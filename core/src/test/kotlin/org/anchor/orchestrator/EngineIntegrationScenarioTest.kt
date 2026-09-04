package org.anchor.orchestrator

import org.anchor.fusion.ErrorStateLayout
import org.anchor.math.Mat
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.math.sin

/**
 * Phase 6: end-to-end scenarios through the ACTUAL EngineOrchestrator,
 * not isolated components -- the point is proving the assembled pipeline
 * survives mode transitions and applies the right update in the right
 * order, not proving any one update's own math (Week 2) or any one
 * scheduling decision in isolation (this PR's other test files).
 *
 * Not attempting real-world navigation accuracy -- these are the four
 * scenarios named in this milestone's own brief, run against synthetic
 * data with independently reasoned-about expected behaviour, same
 * discipline as every synthetic test in this repo since Week 2.
 *
 * NOTE: written, not run via Gradle -- verified via direct kotlinc/
 * junit-platform-console-standalone instead.
 */
class EngineIntegrationScenarioTest {

    private val g = 9.80665
    private val dt = 0.1
    private val tickNanos = (dt * 1_000_000_000L).toLong()

    private fun diagonalCovariance(value: Double = 1.0): Mat =
        Mat.diagonal(DoubleArray(ErrorStateLayout.DIM) { value })

    private fun calibrated(covariance: Mat = diagonalCovariance()): Pair<EngineOrchestrator, Long> {
        val orchestrator = EngineOrchestrator(covariance)
        var timestampNanos = 0L
        repeat(31) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += tickNanos
        }
        return orchestrator to timestampNanos
    }

    @Test
    fun `scenario A -- stationary -- IMU noise, stationarity detection, ZUPT, a stable state`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start

        var stationaryTicks = 0
        val totalTicks = 400
        repeat(totalTicks) { // 40s parked, engine idling
            // Single-axis wobble at ZuptUpdateTest's own already-verified
            // amplitude -- a rolling-window variance metric against an
            // oscillating signal occasionally samples a window skewed
            // toward a peak, so this scenario checks the overwhelming
            // majority classify STATIONARY, not a literal every-tick bar.
            val wobble = 0.002 * sin(it * 0.41)
            val result = orchestrator.onImuSample(Vec3(wobble, 0.0, g), Vec3(0.0004, 0.0, 0.0), timestampNanos)
            assertTrue(result is EngineTickResult.Propagated, "tick $it: expected Propagated, got $result")
            if ((result as EngineTickResult.Propagated).mode == EngineMode.STATIONARY) {
                assertTrue(result.appliedUpdate == "ZUPT")
                stationaryTicks++
            }
            timestampNanos += tickNanos
        }

        assertTrue(stationaryTicks >= totalTicks * 0.9, "$stationaryTicks/$totalTicks ticks classified STATIONARY -- expected the overwhelming majority")
        assertTrue(orchestrator.state.position.norm() < 1.0, "position=${orchestrator.state.position} after 40s idle")
        assertTrue(orchestrator.state.velocity.norm() < 0.5, "velocity=${orchestrator.state.velocity} after 40s idle")
    }

    @Test
    fun `scenario B -- straight driving -- IMU propagation, NHC, forward trajectory with bounded lateral drift`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start

        repeat(300) { // 30s accelerating/cruising in a straight line
            val vibration = 1.5 * sin(it * 1.3)
            val result = orchestrator.onImuSample(Vec3(0.8 + vibration, 0.0, g), Vec3.ZERO, timestampNanos)
            assertTrue(result is EngineTickResult.Propagated && result.mode == EngineMode.DRIVING, "tick $it: expected DRIVING, got $result")
            assertTrue((result as EngineTickResult.Propagated).appliedUpdate == "NHC")
            timestampNanos += tickNanos
        }

        assertTrue(orchestrator.state.position.x > 20.0, "forward position.x=${orchestrator.state.position.x} should reflect 30s of forward acceleration")
        assertTrue(abs(orchestrator.state.position.y) < 5.0, "lateral drift position.y=${orchestrator.state.position.y} should stay bounded under continuous NHC")
        assertTrue(abs(orchestrator.state.position.z) < 5.0, "vertical drift position.z=${orchestrator.state.position.z} should stay bounded under continuous NHC")
    }

    @Test
    fun `scenario C -- GNSS outlier -- normal trajectory, an extreme jump, chi-square rejection, a healthy state`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start

        repeat(100) { // 10s of ordinary driving first
            val vibration = 1.5 * sin(it * 1.3)
            orchestrator.onImuSample(Vec3(0.8 + vibration, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += tickNanos
        }
        val positionBeforeJump = orchestrator.state.position
        val velocityBeforeJump = orchestrator.state.velocity

        val gateResult = orchestrator.applyGnssObservation(Vec3(50_000.0, -50_000.0, 500.0), Vec3(1.0, 1.0, 1.0))
        assertFalse(gateResult.accepted, "a 50km/500m jump must be rejected by the chi-square gate")

        // "State remains healthy": unchanged by the rejected fix, and the
        // pipeline keeps running normally afterward -- not just non-crashing,
        // but numerically continuous with what came before the jump.
        assertTrue(abs(orchestrator.state.position.x - positionBeforeJump.x) < 1e-9)
        assertTrue(abs(orchestrator.state.velocity.x - velocityBeforeJump.x) < 1e-9)

        repeat(50) {
            val vibration = 1.5 * sin(it * 1.3)
            val result = orchestrator.onImuSample(Vec3(0.8 + vibration, 0.0, g), Vec3.ZERO, timestampNanos)
            assertTrue(result is EngineTickResult.Propagated)
            timestampNanos += tickNanos
        }
        assertTrue(orchestrator.state.position.x > positionBeforeJump.x, "the vehicle should keep progressing forward after the rejected outlier")
        assertTrue(orchestrator.state.position.x.isFinite())
    }

    @Test
    fun `scenario D -- GNSS loss -- a valid fix, then unavailability, IMU-NHC-ZUPT continues without crashing or resetting`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start

        repeat(50) { // establish driving
            val vibration = 1.5 * sin(it * 1.3)
            orchestrator.onImuSample(Vec3(0.8 + vibration, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += tickNanos
        }

        // One valid GNSS fix, close to the filter's own estimate -- accepted.
        val plausibleFix = Vec3(orchestrator.state.position.x + 0.5, orchestrator.state.position.y, orchestrator.state.position.z)
        val fixResult = orchestrator.applyGnssObservation(plausibleFix, Vec3(2.0, 2.0, 2.0))
        assertTrue(fixResult.accepted, "statistic=${fixResult.statistic} threshold=${fixResult.threshold}")

        // GNSS now unavailable for a long stretch -- no further
        // applyGnssObservation calls at all (the absence of a call IS the
        // "no fix" signal, matching SensorEvent.GnssFix's own documented
        // convention). Only IMU ticks, driving then parking, exercising
        // both NHC and ZUPT with no GNSS in the loop.
        repeat(200) { // 20s more driving
            val vibration = 1.5 * sin(it * 1.3)
            val result = orchestrator.onImuSample(Vec3(0.8 + vibration, 0.0, g), Vec3.ZERO, timestampNanos)
            assertTrue(result is EngineTickResult.Propagated)
            timestampNanos += tickNanos
        }
        var stationaryTicks = 0
        val parkedTicks = 200
        repeat(parkedTicks) { // 20s parked afterward
            val wobble = 0.002 * sin(it * 0.41)
            val result = orchestrator.onImuSample(Vec3(wobble, 0.0, g), Vec3.ZERO, timestampNanos)
            assertTrue(result is EngineTickResult.Propagated)
            if ((result as EngineTickResult.Propagated).mode == EngineMode.STATIONARY) stationaryTicks++
            timestampNanos += tickNanos
        }

        assertTrue(orchestrator.state.position.x.isFinite() && orchestrator.state.velocity.norm().isFinite(), "40s of GNSS-less operation must not crash or produce NaN")
        assertTrue(stationaryTicks >= parkedTicks * 0.9, "$stationaryTicks/$parkedTicks parked ticks classified STATIONARY -- should have settled back once parked")
    }
}
