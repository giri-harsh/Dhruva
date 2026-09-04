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
 * Phase 3: does the ORCHESTRATOR correctly decide, without being told,
 * that the vehicle is driving and NHC should run -- as opposed to Week
 * 2's NhcUpdateTest, which already covers whether NHC's own math is
 * correct once a caller manually decides to invoke it.
 *
 * A driving-classification subtlety these tests rely on: StationarityDetector
 * measures FLUCTUATION, not the mean, so a perfectly constant acceleration
 * (however large) alone reads as "stationary" -- zero variance. Every
 * scenario below rides realistic road-vibration-scale noise on top of the
 * commanded motion (StationarityDetectorTest's own already-verified
 * "sustained road-vibration" magnitude) specifically to keep the engine
 * in DRIVING mode throughout, not just during an initial ramp-up.
 *
 * NOTE: written, not run via Gradle -- verified via direct kotlinc/
 * junit-platform-console-standalone instead.
 */
class EngineOrchestratorNhcTest {

    private val g = 9.80665
    private val dt = 0.1

    private fun diagonalCovariance(value: Double = 0.01): Mat =
        Mat.diagonal(DoubleArray(ErrorStateLayout.DIM) { value })

    private fun calibrated(covariance: Mat = diagonalCovariance()): Pair<EngineOrchestrator, Long> {
        val orchestrator = EngineOrchestrator(covariance)
        var timestampNanos = 0L
        repeat(31) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        return orchestrator to timestampNanos
    }

    /** Road-vibration-scale accel wobble, matching
     *  StationarityDetectorTest's own "not classified as stationary" fixture. */
    private fun drivingWobble(i: Int): Double = 1.5 * sin(i * 1.3)

    @Test
    fun `lateral velocity drift is pulled toward zero once the engine is driving`() {
        // Inject the drift as an INITIAL condition rather than a transient
        // accel pulse: NHC runs every driving tick, including during a
        // pulse, so a pulse's own body-lateral peak is already partly
        // corrected by the time it ends, muddying a before/after
        // comparison. Starting with the drift already in state.velocity
        // sidesteps that -- and since orientation starts at IDENTITY and
        // calibration never touches state.velocity, nav-frame Y == body-
        // frame Y exactly at t=0, so this is unambiguous at the start.
        val initialState = org.anchor.fusion.NominalState.zero().copy(velocity = Vec3(0.0, 3.0, 0.0))
        val orchestrator = EngineOrchestrator(diagonalCovariance(), initialState = initialState)
        var timestampNanos = 0L
        repeat(31) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }

        val lateralAtStart = abs(orchestrator.state.bodyVelocity().y)
        assertTrue(lateralAtStart > 1.0, "initial condition should carry real lateral drift: $lateralAtStart")

        var driving = false
        repeat(40) {
            val result = orchestrator.onImuSample(Vec3(1.0 + drivingWobble(it), 0.0, g), Vec3.ZERO, timestampNanos)
            if (result is EngineTickResult.Propagated) driving = driving || result.mode == EngineMode.DRIVING
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        assertTrue(driving, "forward acceleration with realistic vibration should have entered DRIVING")

        val lateralAfterCorrection = abs(orchestrator.state.bodyVelocity().y)
        assertTrue(
            lateralAfterCorrection < lateralAtStart,
            "BODY-frame lateral velocity should shrink under repeated automatic NHC: $lateralAtStart -> $lateralAfterCorrection",
        )
    }

    @Test
    fun `vertical velocity drift is pulled toward zero once the engine is driving`() {
        // Same initial-condition technique as the lateral test above, and
        // for the same reason: NHC runs every driving tick, so a transient
        // pulse is already partly self-correcting by the time it ends.
        val initialState = org.anchor.fusion.NominalState.zero().copy(velocity = Vec3(0.0, 0.0, 3.0))
        val orchestrator = EngineOrchestrator(diagonalCovariance(), initialState = initialState)
        var timestampNanos = 0L
        repeat(31) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }

        val verticalAtStart = abs(orchestrator.state.bodyVelocity().z)
        assertTrue(verticalAtStart > 1.0, "initial condition should carry real vertical drift: $verticalAtStart")

        repeat(40) {
            orchestrator.onImuSample(Vec3(1.0 + drivingWobble(it), 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }

        val verticalAfterCorrection = abs(orchestrator.state.bodyVelocity().z)
        assertTrue(
            verticalAfterCorrection < verticalAtStart,
            "BODY-frame vertical velocity should shrink under repeated automatic NHC: $verticalAtStart -> $verticalAfterCorrection",
        )
    }

    @Test
    fun `forward vehicle motion remains unconstrained while NHC is auto-dispatched`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start
        repeat(50) {
            orchestrator.onImuSample(Vec3(1.0 + drivingWobble(it), 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        // 50 ticks * 0.1s at ~1 m/s^2 forward: NHC must not have suppressed
        // this the way it suppresses y/z -- forward speed should be
        // substantial, not driven toward zero.
        assertTrue(orchestrator.state.velocity.x > 2.0, "forward velocity.x=${orchestrator.state.velocity.x} should not be constrained by NHC")
    }

    @Test
    fun `NHC constrains BODY-frame velocity correctly through a sustained heading change`() {
        val (orchestrator, start) = calibrated()
        var timestampNanos = start
        val yawRate = 0.1 // rad/s, constant -- a steady turn

        // Get driving first (straight, no yaw) so classification is established.
        repeat(15) {
            orchestrator.onImuSample(Vec3(1.0 + drivingWobble(it), 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }

        // Now turn: constant forward accel + constant yaw rate + the same
        // accel vibration (constant yaw rate alone has zero variance and
        // would not, by itself, keep StationarityDetector's gyro-energy
        // check from reading near-zero -- the accel wobble is what keeps
        // this tick classified as DRIVING).
        val bodyLateralSamples = ArrayList<Double>()
        val bodyVerticalSamples = ArrayList<Double>()
        repeat(150) { // 15s at 0.1 rad/s ~= 1.5 rad (~86 degrees) of turn
            val result = orchestrator.onImuSample(Vec3(1.0 + drivingWobble(it), 0.0, g), Vec3(0.0, 0.0, yawRate), timestampNanos)
            assertTrue(result is EngineTickResult.Propagated && result.mode == EngineMode.DRIVING, "expected DRIVING at tick $it, got $result")
            if (it % 10 == 0) {
                val bodyVel = orchestrator.state.bodyVelocity()
                bodyLateralSamples.add(abs(bodyVel.y))
                bodyVerticalSamples.add(abs(bodyVel.z))
            }
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }

        // The key claim: body-frame lateral/vertical stayed small THROUGHOUT
        // the turn, not just at the start -- proof the Jacobian is correctly
        // re-evaluated against the CURRENT (rotating) orientation each tick,
        // not a fixed one from before the turn began.
        for ((i, lateral) in bodyLateralSamples.withIndex()) {
            assertTrue(lateral < 1.0, "body-frame lateral velocity=$lateral too large at sample $i during the turn")
        }
        for ((i, vertical) in bodyVerticalSamples.withIndex()) {
            assertTrue(vertical < 1.0, "body-frame vertical velocity=$vertical too large at sample $i during the turn")
        }
        // And the turn actually happened: nav-frame velocity direction should
        // have rotated away from pure +x as heading changed.
        assertTrue(abs(orchestrator.state.velocity.y) > 0.5, "nav-frame velocity.y=${orchestrator.state.velocity.y} should reflect the vehicle having turned")
    }
}
