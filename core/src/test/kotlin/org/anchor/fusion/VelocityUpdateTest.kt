package org.anchor.fusion

import org.anchor.math.Mat
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

/** NOTE: written, not run via Gradle -- see the Week-1 report for why. */
class VelocityUpdateTest {

    @Test
    fun `noise is exactly sigma squared times trust factor, FR-11 own formula`() {
        val update = VelocityUpdate(meanMps = 5.0, variance = 0.3, trustFactor = 2.0)
        assertEquals(0.6, update.noise()[0, 0], 1e-12)
    }

    @Test
    fun `predicted reads the forward, x-axis, body velocity component`() {
        val state = NominalState.zero().copy(velocity = org.anchor.math.Vec3(7.0, 1.0, 0.0))
        val predicted = VelocityUpdate(meanMps = 0.0, variance = 1.0).predicted(state)
        assertEquals(7.0, predicted[0], 1e-9)
    }

    @Test
    fun `doubling sigma squared halves the state correction magnitude, FR-11 explicit acceptance test`() {
        // Constructed deliberately in the R-dominates-S regime (see this
        // class's own doc comment for why that is the meaningful regime,
        // not a special case chosen to force the result): P[vel_x] is
        // tight (a well-converged filter), R is comparatively large (an
        // uncertain learned-velocity reading), so S ~= R and K ~= P/R,
        // making the correction scale as 1/R almost exactly.
        val tightVelocityVariance = 1e-4
        val diag = DoubleArray(ErrorStateLayout.DIM) { 1e-6 }
        diag[ErrorStateLayout.VEL] = tightVelocityVariance
        val covariance = Mat.diagonal(diag)

        val baseVariance = 1.0
        val ekfLowVariance = ErrorStateEkf(NominalState.zero(), covariance.copy())
        ekfLowVariance.correct(VelocityUpdate(meanMps = 10.0, variance = baseVariance))
        val correctionLow = ekfLowVariance.state.velocity.x

        val ekfHighVariance = ErrorStateEkf(NominalState.zero(), covariance.copy())
        ekfHighVariance.correct(VelocityUpdate(meanMps = 10.0, variance = baseVariance * 2.0))
        val correctionHigh = ekfHighVariance.state.velocity.x

        val ratio = correctionHigh / correctionLow
        assertEquals(0.5, ratio, 0.02, "doubling variance should halve the correction; got ratio $ratio")
    }

    @Test
    fun `a slower-than-reported forward speed pulls velocity down toward the measurement`() {
        val state = NominalState.zero().copy(velocity = org.anchor.math.Vec3(15.0, 0.0, 0.0))
        val diag = DoubleArray(ErrorStateLayout.DIM) { 0.5 }
        val ekf = ErrorStateEkf(state, Mat.diagonal(diag))
        ekf.correct(VelocityUpdate(meanMps = 10.0, variance = 0.5))
        assertEquals(true, ekf.state.velocity.x < 15.0)
        assertEquals(true, ekf.state.velocity.x > 10.0) // partial correction, not a snap to the measurement
    }
}
