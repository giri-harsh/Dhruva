package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** NOTE: written, not run via Gradle -- see the Week-1 report for why. */
class NhcUpdateTest {

    private fun covarianceWithVelocityUncertainty(velVar: Double = 1.0): Mat {
        val diag = DoubleArray(ErrorStateLayout.DIM) { 0.01 }
        for (i in 0 until 3) diag[ErrorStateLayout.VEL + i] = velVar
        return Mat.diagonal(diag)
    }

    @Test
    fun `predicted extracts lateral and vertical body velocity, forward axis excluded`() {
        // Orientation is identity, so body frame == nav frame here: a
        // sideways+vertical drift with zero forward speed.
        val state = NominalState.zero().copy(velocity = Vec3(0.0, 0.7, -0.3))
        val predicted = NhcUpdate().predicted(state)
        assertEquals(0.7, predicted[0], 1e-9)
        assertEquals(-0.3, predicted[1], 1e-9)
    }

    @Test
    fun `actual target is always zero, the non-holonomic assumption itself`() {
        val actual = NhcUpdate().actual()
        assertEquals(0.0, actual[0], 1e-9)
        assertEquals(0.0, actual[1], 1e-9)
    }

    @Test
    fun `a lateral velocity drift is corrected toward zero by an NHC update`() {
        val state = NominalState.zero().copy(velocity = Vec3(5.0, 0.5, 0.0)) // forward + spurious lateral
        val ekf = ErrorStateEkf(state, covarianceWithVelocityUncertainty())
        ekf.correct(NhcUpdate())
        // Forward component should be left materially intact (NHC does not
        // observe it); lateral should move toward zero.
        assertTrue(ekf.state.velocity.y < 0.5, "lateral velocity should shrink toward zero after NHC")
        assertTrue(ekf.state.velocity.x > 3.0, "NHC should not have zeroed out the untouched forward component")
    }

    @Test
    fun `jacobian has zero columns for position and bias blocks, only velocity and theta contribute`() {
        val state = NominalState.zero().copy(velocity = Vec3(4.0, 0.1, -0.1))
        val h = NhcUpdate().jacobian(state)
        assertEquals(2, h.rows)
        assertEquals(ErrorStateLayout.DIM, h.cols)
        for (i in 0 until 2) {
            for (c in ErrorStateLayout.POS until ErrorStateLayout.POS + ErrorStateLayout.BLOCK) assertEquals(0.0, h[i, c], 1e-12)
            for (c in ErrorStateLayout.ACCEL_BIAS until ErrorStateLayout.GYRO_BIAS + ErrorStateLayout.BLOCK) assertEquals(0.0, h[i, c], 1e-12)
        }
    }
}
