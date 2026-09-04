package org.anchor.orchestrator

import org.anchor.fusion.ErrorStateLayout
import org.anchor.math.Mat
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Covers the two things that must be true before ANY other orchestrator
 * behaviour can be trusted: (1) the engine stays in CALIBRATING and never
 * touches the EKF until FR-04's own 3s window elapses, and (2) the
 * gravity sign-reconciliation this class's own doc comment describes
 * (GravityAlign's rotation, built for the gravity-REMOVED ML consumer,
 * re-derived here with the correct sign for the gravity-INCLUDED ESKF
 * consumer) actually produces a body-frame reading that leaves a
 * level, stationary vehicle at zero net acceleration -- not silently
 * doubled or inverted. If this test is wrong, every other test in this
 * PR that calibrates first and then asserts on physical behaviour is
 * unknowingly built on a broken foundation.
 *
 * NOTE: written, not run via Gradle -- see the Week-1 report for why;
 * verified via direct kotlinc/junit-platform-console-standalone instead.
 */
class EngineOrchestratorCalibrationTest {

    private val g = 9.80665
    private val dt = 0.1
    private val ticksFor3s = 31 // 31 * 0.1s = 3.1s, clears FR-04's 3s window with margin

    private fun diagonalCovariance(value: Double = 0.01): Mat =
        Mat.diagonal(DoubleArray(ErrorStateLayout.DIM) { value })

    @Test
    fun `stays in CALIBRATING and leaves the EKF untouched before the 3s window elapses`() {
        val orchestrator = EngineOrchestrator(diagonalCovariance())
        var timestampNanos = 0L
        repeat(20) { // 20 * 0.1s = 2.0s, short of the 3s window
            val result = orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            assertTrue(result is EngineTickResult.Calibrating, "expected Calibrating, got $result")
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        assertEquals(EngineMode.CALIBRATING, orchestrator.mode)
        assertEquals(0.0, orchestrator.state.position.x, 1e-12); assertEquals(0.0, orchestrator.state.position.y, 1e-12); assertEquals(0.0, orchestrator.state.position.z, 1e-12)
        assertEquals(0.0, orchestrator.state.velocity.x, 1e-12); assertEquals(0.0, orchestrator.state.velocity.y, 1e-12); assertEquals(0.0, orchestrator.state.velocity.z, 1e-12)
    }

    @Test
    fun `transitions out of CALIBRATING once the 3s window elapses`() {
        val orchestrator = EngineOrchestrator(diagonalCovariance())
        var timestampNanos = 0L
        var lastResult: EngineTickResult? = null
        repeat(ticksFor3s) {
            lastResult = orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        assertTrue(lastResult is EngineTickResult.Calibrating, "the boundary tick itself still reports Calibrating")

        val nextResult = orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
        assertTrue(nextResult is EngineTickResult.Propagated, "expected Propagated once past the window, got $nextResult")
        assertTrue(orchestrator.mode != EngineMode.CALIBRATING)
    }

    @Test
    fun `gravity sign reconciliation -- a level stationary phone stays at zero velocity after calibration`() {
        // Phone-frame reading matches contracts/frame_convention.md line 16
        // and ml features.py exactly: accel_z ~= +9.8 at rest, gravity-
        // inclusive, raw. If EngineOrchestrator's negated-gravity fix in
        // freezeEskfRotation() were wrong or missing, the resulting
        // rotation would hand propagate() a NEGATIVE body-z reading,
        // which GRAVITY_NAV would then double rather than cancel --
        // velocity would run away at roughly 2g instead of staying zero.
        val orchestrator = EngineOrchestrator(diagonalCovariance())
        var timestampNanos = 0L
        repeat(ticksFor3s + 5) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        // A handful more RUNNING-phase ticks of the identical stationary reading.
        repeat(20) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }

        assertTrue(
            kotlin.math.abs(orchestrator.state.velocity.z) < 0.5,
            "velocity.z=${orchestrator.state.velocity.z} -- a level, stationary phone must not accelerate " +
                "under its own (correctly cancelled) weight; a value near +/-2g here means the gravity " +
                "sign reconciliation regressed",
        )
        assertTrue(kotlin.math.abs(orchestrator.state.position.z) < 2.0, "position.z=${orchestrator.state.position.z}")
    }

    @Test
    fun `gravity sign reconciliation holds for a non-trivial mounting too, not only the antiparallel special case`() {
        // Phone mounted so gravity reads on its OWN x-axis (GravityAlignTest's
        // own "general Rodrigues branch" fixture: Vec3(9.8, 0, 0)) rather than
        // z -- exercises the general rotation path, not just diag(1,-1,-1).
        val orchestrator = EngineOrchestrator(diagonalCovariance())
        var timestampNanos = 0L
        repeat(ticksFor3s + 5) {
            orchestrator.onImuSample(Vec3(g, 0.0, 0.0), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        repeat(20) {
            orchestrator.onImuSample(Vec3(g, 0.0, 0.0), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }

        val speed = orchestrator.state.velocity.norm()
        assertTrue(speed < 1.0, "velocity norm=$speed -- a stationary phone in a different mounting orientation must also settle near zero")
    }
}
