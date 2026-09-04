package org.anchor.orchestrator

import org.anchor.fusion.ErrorStateLayout
import org.anchor.math.Mat
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Phase 5: the GNSS pipeline as wired into EngineOrchestrator --
 * observation -> GatedMeasurementUpdate (innovation, S, chi-square gate)
 * -> accepted/rejected update. Deliberately does not touch geodetic
 * conversion: every observation here is already local-ENU, exactly
 * GnssPositionUpdate's own existing contract (see EngineOrchestrator's
 * applyGnssObservation doc). No map matching, no lat/lon.
 *
 * NOTE: written, not run via Gradle -- verified via direct kotlinc/
 * junit-platform-console-standalone instead.
 */
class EngineOrchestratorGnssTest {

    private val g = 9.80665
    private val dt = 0.1

    private fun diagonalCovariance(value: Double = 4.0): Mat =
        Mat.diagonal(DoubleArray(ErrorStateLayout.DIM) { value })

    private fun calibrated(covariance: Mat = diagonalCovariance()): EngineOrchestrator {
        val orchestrator = EngineOrchestrator(covariance)
        var timestampNanos = 0L
        repeat(31) {
            orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        return orchestrator
    }

    @Test
    fun `a GNSS observation before calibration completes is refused rather than silently misapplied`() {
        val orchestrator = EngineOrchestrator(diagonalCovariance())
        assertThrows(IllegalStateException::class.java) {
            orchestrator.applyGnssObservation(Vec3(10.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
        }
    }

    @Test
    fun `a plausible local-ENU GNSS fix is accepted and pulls position toward it`() {
        val orchestrator = calibrated()
        val positionBefore = orchestrator.state.position.x

        val result = orchestrator.applyGnssObservation(Vec3(2.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))

        assertTrue(result.accepted, "statistic=${result.statistic} threshold=${result.threshold}")
        assertTrue(orchestrator.state.position.x > positionBefore, "an accepted fix should pull position toward the observation")
    }

    @Test
    fun `an extreme GNSS outlier is rejected and the state is left healthy`() {
        val orchestrator = calibrated()
        val positionBefore = orchestrator.state.position
        val covarianceBefore = orchestrator.covariance[ErrorStateLayout.POS, ErrorStateLayout.POS]

        // 10 km away -- nowhere near plausible against a starting covariance of 4.
        val result = orchestrator.applyGnssObservation(Vec3(10_000.0, 10_000.0, 0.0), Vec3(1.0, 1.0, 1.0))

        assertFalse(result.accepted, "a 10km jump must be rejected, not fused")
        assertEquals(positionBefore.x, orchestrator.state.position.x, 1e-9)
        assertEquals(positionBefore.y, orchestrator.state.position.y, 1e-9)
        assertEquals(positionBefore.z, orchestrator.state.position.z, 1e-9)
        assertEquals(covarianceBefore, orchestrator.covariance[ErrorStateLayout.POS, ErrorStateLayout.POS], 1e-9)
    }

    @Test
    fun `IMU propagation continues normally after a rejected GNSS fix -- no crash, no reset`() {
        val orchestrator = calibrated()
        orchestrator.applyGnssObservation(Vec3(10_000.0, 10_000.0, 0.0), Vec3(1.0, 1.0, 1.0))

        var timestampNanos = 31L * (dt * 1_000_000_000L).toLong()
        repeat(20) {
            val result = orchestrator.onImuSample(Vec3(0.0, 0.0, g), Vec3.ZERO, timestampNanos)
            assertTrue(result is EngineTickResult.Propagated)
            timestampNanos += (dt * 1_000_000_000L).toLong()
        }
        assertTrue(orchestrator.state.position.x.isFinite() && orchestrator.state.velocity.x.isFinite())
    }
}
