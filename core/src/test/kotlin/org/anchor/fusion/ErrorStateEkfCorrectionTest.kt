package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Phase 3.E: generic measurement-update validation. Uses a trivial,
 * hand-defined MeasurementModel (directly observes position.x) rather
 * than NHC/ZUPT/VelocityUpdate, so this test is decoupled from any
 * specific physical model's own correctness -- it is purely about
 * ErrorStateEkf.correct()'s generic machinery: innovation, innovation
 * covariance, Kalman gain, state correction, Joseph-form covariance.
 *
 * NOTE: written, not run via Gradle -- see the Week-1 report for why.
 */
class ErrorStateEkfCorrectionTest {

    /** Directly observes position.x -- the simplest possible non-trivial
     *  MeasurementModel, H = [1,0,...,0] (1x15), used only by this test. */
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
    fun `innovation is the difference between actual and predicted measurement`() {
        val state = NominalState.zero().copy(position = Vec3(3.0, 0.0, 0.0))
        val ekf = ErrorStateEkf(state, diagonalCovariance())
        val measurement = DirectPositionXObservation(observed = 5.0, varianceValue = 1.0)

        // predicted() = 3.0 (current state), actual() = 5.0 -> innovation = 2.0.
        // Verified indirectly via the resulting correction below (the class
        // does not expose innovation directly; that is a deliberate,
        // minimal public surface -- state/covariance are the contract).
        val positionBefore = ekf.state.position.x
        ekf.correct(measurement)
        val positionAfter = ekf.state.position.x

        // With posVar=4, R=1: K = P/(P+R) = 4/5 = 0.8; correction = 0.8*2.0 = 1.6
        assertEquals(positionBefore + 1.6, positionAfter, 1e-9)
    }

    @Test
    fun `Kalman gain and correction match a hand-computed 1D case exactly`() {
        // 1D case is fully hand-checkable: K = P / (P+R), correction = K * innovation.
        val posVar = 9.0
        val r = 3.0
        val observed = 10.0
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance(posVar))

        ekf.correct(DirectPositionXObservation(observed, r))

        val expectedK = posVar / (posVar + r) // 9/12 = 0.75
        val expectedCorrection = expectedK * observed // innovation = 10 - 0 = 10
        assertEquals(expectedCorrection, ekf.state.position.x, 1e-9)

        // Joseph form in 1D: P' = (1-K)^2 * P + K^2 * R
        val expectedP = (1 - expectedK) * (1 - expectedK) * posVar + expectedK * expectedK * r
        assertEquals(expectedP, ekf.covariance[ErrorStateLayout.POS, ErrorStateLayout.POS], 1e-9)
    }

    @Test
    fun `covariance stays symmetric and positive-diagonal after many repeated corrections`() {
        // FR-09's own acceptance criterion, exercised on the update path
        // rather than only propagation.
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance())
        repeat(200) {
            ekf.correct(DirectPositionXObservation(observed = (it % 5) * 0.1, varianceValue = 0.5))
            for (i in 0 until ErrorStateLayout.DIM) {
                assertTrue(ekf.covariance[i, i] > 0.0, "P[$i,$i] went non-positive after ${it + 1} corrections")
                for (j in 0 until ErrorStateLayout.DIM) {
                    assertEquals(ekf.covariance[i, j], ekf.covariance[j, i], 1e-9, "asymmetric at ($i,$j)")
                }
            }
        }
    }

    @Test
    fun `a correction reduces the corrected state's own variance, information is gained`() {
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance(posVar = 4.0))
        val before = ekf.covariance[ErrorStateLayout.POS, ErrorStateLayout.POS]
        ekf.correct(DirectPositionXObservation(observed = 1.0, varianceValue = 1.0))
        val after = ekf.covariance[ErrorStateLayout.POS, ErrorStateLayout.POS]
        assertTrue(after < before, "a measurement update should reduce the observed state's variance")
    }

    @Test
    fun `correct rejects a jacobian with the wrong shape rather than silently misusing it`() {
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance())
        val malformed = object : MeasurementModel {
            override val name = "Malformed"
            override val dimension = 1
            override fun predicted(state: NominalState) = Vec.of(0.0)
            override fun actual() = Vec.of(0.0)
            override fun jacobian(state: NominalState) = Mat.zeros(2, ErrorStateLayout.DIM) // wrong: dimension says 1
            override fun noise() = Mat(1, 1, doubleArrayOf(1.0))
        }
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException::class.java) { ekf.correct(malformed) }
    }
}
