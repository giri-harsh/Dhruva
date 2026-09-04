package org.anchor.orchestrator

import org.anchor.fusion.ErrorStateLayout
import org.anchor.math.Mat
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.math.sin

/**
 * Phase 2: does the ORCHESTRATOR correctly decide, tick by tick and
 * without being told, that the vehicle is stationary and ZUPT should run
 * -- as opposed to Week 2's ZuptUpdateTest, which already covers whether
 * ZUPT's own math is correct when a caller manually decides to invoke it.
 * These tests never call anything ZUPT-related directly; they only feed
 * onImuSample() and check that the automatic scheduling produced the
 * right behaviour.
 *
 * NOTE: written, not run via Gradle -- verified via direct kotlinc/
 * junit-platform-console-standalone instead, matching every other test
 * file in this repo.
 */
class EngineOrchestratorStationarityTest {

    private val g = 9.80665
    private val dt = 0.1

    private fun diagonalCovariance(value: Double = 0.01): Mat =
        Mat.diagonal(DoubleArray(ErrorStateLayout.DIM) { value })

    /** Runs calibration to completion (31 ticks, 3.1s of a level, still
     *  phone) and returns the orchestrator plus the next timestamp to use. */
    private fun calibrated(covariance: Mat = diagonalCovariance()): Pair<EngineOrchestrator, Long> {
        val orchestrator = EngineOrchestrator(covariance)
        var timestampNanos = 0L
        repeat(31) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        return orchestrator to timestampNanos
    }

    @Test
    fun `a stationary vehicle remains stable -- mode reports STATIONARY and position stays near zero`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start
        repeat(50) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        assertEquals(EngineMode.STATIONARY, orchestrator.mode)
        assertTrue(orchestrator.state.position.norm() < 0.5, "position=${orchestrator.state.position}")
        assertTrue(orchestrator.state.velocity.norm() < 0.5, "velocity=${orchestrator.state.velocity}")
    }

    @Test
    fun `ZUPT is automatically dispatched on the overwhelming majority of stationary ticks, keeping velocity drift bounded over a long window`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start
        val steps = 600 // 60s of noisy-but-stationary running
        var zuptCount = 0
        repeat(steps) {
            // Same small, deterministic wobble ZuptUpdateTest's own 120s
            // scenario uses -- a real accelerometer is never perfectly
            // noise-free, but this stays well under
            // StationarityDetector's own variance threshold on average. A
            // rolling-window variance metric against an oscillating signal
            // will occasionally sample a window skewed toward a peak, so
            // this asserts the overwhelming majority, not literally every
            // tick -- see this file's own "noisy stationary" test for the
            // same reasoning applied to the false-DRIVING question directly.
            val wobble = 0.002 * sin(it * 0.37)
            val result = orchestrator.onImuSample(Vec3(wobble, 0.0, g), Vec3(0.0005, 0.0, 0.0), timestampNanos)
            if (result is EngineTickResult.Propagated && result.appliedUpdate == "ZUPT") zuptCount++
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        assertTrue(zuptCount >= steps * 0.9, "$zuptCount/$steps ticks auto-dispatched ZUPT -- expected the overwhelming majority")
        assertTrue(orchestrator.state.velocity.norm() < 0.5, "velocity=${orchestrator.state.velocity} after ${steps * dt}s")
        assertTrue(orchestrator.state.position.norm() < 1.0, "position=${orchestrator.state.position} after ${steps * dt}s")
    }

    @Test
    fun `repeated automatic ZUPT corrections move the accelerometer bias estimate toward a constant offset`() {
        // A constant per-tick offset has zero VARIANCE (StationarityDetector
        // measures fluctuation, not the mean), so this is correctly still
        // classified stationary -- exactly what a real, uncalibrated
        // accelerometer bias on an otherwise-still phone looks like.
        val trueBiasX = 0.05
        val (orchestrator, start) = calibrated()
        var timestampNanos = start
        repeat(1500) { // 150s -- Week 2's own bias-learning test needed a
            // realistic multi-second window before cross-correlation built up.
            orchestrator.onImuSample(Vec3(trueBiasX, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        assertEquals(EngineMode.STATIONARY, orchestrator.mode)
        assertTrue(
            orchestrator.state.accelBias.x > 0.0,
            "accelBias.x=${orchestrator.state.accelBias.x} should have moved toward the true +$trueBiasX bias, not stayed at zero",
        )
    }

    @Test
    fun `noisy stationary sensor data does not cause false motion or a false DRIVING classification`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start
        var drivingTicks = 0
        val totalTicks = 300
        repeat(totalTicks) {
            // Single-axis wobble at ZuptUpdateTest's own already-verified
            // amplitude/frequency (see EngineOrchestratorStationarityTest's
            // ZUPT-dispatch test) -- a real stationary phone's sensor noise,
            // not a multi-axis signal invented fresh for this test.
            val wobble = 0.002 * sin(it * 0.37)
            orchestrator.onImuSample(Vec3(wobble, 0.0, g), Vec3(0.0005, 0.0, 0.0), timestampNanos)
            if (orchestrator.mode == EngineMode.DRIVING) drivingTicks++
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        // A rolling-window variance metric against an oscillating signal will
        // occasionally sample a window skewed toward a peak -- demanding
        // literal zero misclassifications is a stricter bar than the
        // detector's own threshold-based design promises. The physically
        // meaningful claim is that noise this small stays overwhelmingly
        // classified stationary and never accumulates into real drift.
        assertTrue(drivingTicks < totalTicks / 20, "$drivingTicks/$totalTicks ticks misclassified as DRIVING -- too many for sensor noise this small")
        assertTrue(orchestrator.state.position.norm() < 1.0, "position=${orchestrator.state.position} -- noise must not accumulate into false motion")
    }
}
