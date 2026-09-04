package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Phase 4: wires ChiSquareGate into the actual measurement-update path.
 * Reuses ErrorStateEkfCorrectionTest's own DirectPositionXObservation
 * pattern (a trivial, hand-defined MeasurementModel decoupled from any
 * concrete update's own correctness) so this stays about
 * GatedMeasurementUpdate's own logic -- accept/reject/state-preservation
 * -- not about NHC/ZUPT/GNSS's math.
 *
 * NOTE: written, not run via Gradle -- verified via direct kotlinc/
 * junit-platform-console-standalone instead.
 */
class GatedMeasurementUpdateTest {

    private class DirectPositionXObservation(private val observed: Double, private val varianceValue: Double) : MeasurementModel {
        override val name = "TestDirectPositionX"
        override val dimension = 1
        override fun predicted(state: NominalState): Vec = Vec.of(state.position.x)
        override fun actual(): Vec = Vec.of(observed)
        override fun jacobian(state: NominalState): Mat {
            val h = Mat.zeros(1, ErrorStateLayout.DIM)
            h[0, ErrorStateLayout.POS] = 1.0
            return h
        }
        override fun noise(): Mat = Mat(1, 1, doubleArrayOf(varianceValue))
    }

    private fun diagonalCovariance(posVar: Double = 4.0): Mat {
        val diag = DoubleArray(ErrorStateLayout.DIM) { 1.0 }
        diag[ErrorStateLayout.POS] = posVar
        return Mat.diagonal(diag)
    }

    @Test
    fun `a plausible measurement is accepted and actually applied`() {
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance(posVar = 4.0))
        // std = 2 (posVar=4), a 1.0 deviation is unremarkable.
        val result = GatedMeasurementUpdate.apply(ekf, DirectPositionXObservation(observed = 1.0, varianceValue = 1.0))
        assertTrue(result.accepted, "statistic=${result.statistic} threshold=${result.threshold}")
        assertTrue(ekf.state.position.x != 0.0, "an accepted measurement must actually have been applied to the state")
    }

    @Test
    fun `an extreme outlier is rejected, FR-27's own acceptance test at the orchestration layer`() {
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance(posVar = 4.0))
        val result = GatedMeasurementUpdate.apply(ekf, DirectPositionXObservation(observed = 500.0, varianceValue = 1.0))
        assertFalse(result.accepted, "a 500-unit jump against posVar=4 must be rejected")
        assertTrue(result.statistic > result.threshold)
    }

    @Test
    fun `a rejected measurement does not corrupt state or covariance`() {
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance(posVar = 4.0))
        val covarianceBefore = ekf.covariance[ErrorStateLayout.POS, ErrorStateLayout.POS]

        val result = GatedMeasurementUpdate.apply(ekf, DirectPositionXObservation(observed = 500.0, varianceValue = 1.0))

        assertFalse(result.accepted)
        assertEquals(0.0, ekf.state.position.x, 1e-12, "a rejected measurement must leave the state exactly as it was")
        assertEquals(covarianceBefore, ekf.covariance[ErrorStateLayout.POS, ErrorStateLayout.POS], 1e-12, "a rejected measurement must leave covariance exactly as it was")
    }

    @Test
    fun `the gate decision -- statistic, threshold, and accepted flag -- is observable to the caller`() {
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance(posVar = 4.0))
        val accepted = GatedMeasurementUpdate.apply(ekf, DirectPositionXObservation(observed = 1.0, varianceValue = 1.0))
        val rejected = GatedMeasurementUpdate.apply(ekf, DirectPositionXObservation(observed = 500.0, varianceValue = 1.0))

        // Both outcomes hand back a fully-populated, inspectable Result --
        // nothing about the gate's reasoning is hidden inside this object.
        assertTrue(accepted.statistic >= 0.0 && accepted.threshold > 0.0)
        assertTrue(rejected.statistic >= 0.0 && rejected.threshold > 0.0)
        assertTrue(accepted.accepted && !rejected.accepted)
    }
}
