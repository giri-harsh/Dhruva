package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec

/**
 * FR-26: zero-velocity update. Unlike NHC/VelocityUpdate (which observe a
 * BODY-frame velocity component and need BodyVelocityJacobian's rotation-
 * dependent derivation), ZUPT observes NAV-frame velocity directly --
 * "the vehicle is not moving" makes no reference to its own orientation,
 * so the Jacobian is the simplest possible: H = [0, I, 0, 0, 0], a direct
 * observation of the delta-v error block.
 *
 * FR-26 also requires ZUPT to "re-observe" accelerometer and gyroscope
 * bias. No special-cased bias-observation term is needed for that: the
 * standard Kalman math already does it, because propagate() has been
 * building up covariance cross-correlation between delta-v and the bias
 * blocks (via F_c's off-diagonal terms) since the last correction -- once
 * that correlation exists, a velocity-only measurement update naturally
 * moves the correlated bias estimates too, through the Kalman gain.
 */
class ZuptUpdate(
    private val velocityNoiseVariance: Double = DEFAULT_NOISE_VARIANCE,
) : MeasurementModel {
    override val name: String = "ZUPT"
    override val dimension: Int = 3

    override fun predicted(state: NominalState): Vec =
        Vec.of(state.velocity.x, state.velocity.y, state.velocity.z)

    override fun actual(): Vec = Vec.of(0.0, 0.0, 0.0)

    override fun jacobian(state: NominalState): Mat {
        val h = Mat.zeros(3, ErrorStateLayout.DIM)
        for (i in 0 until 3) h[i, ErrorStateLayout.VEL + i] = 1.0
        return h
    }

    override fun noise(): Mat = Mat.diagonal(doubleArrayOf(velocityNoiseVariance, velocityNoiseVariance, velocityNoiseVariance))

    companion object {
        /** (m/s)^2 -- small but nonzero: a real "stationary" vehicle still
         *  has minor vibration; a variance of exactly 0 would make ZUPT
         *  claim perfect certainty, which FR-28's "a wrong update can
         *  never make the filter certain" principle (stated for map
         *  updates, equally true here) argues against. [VERIFY]. */
        const val DEFAULT_NOISE_VARIANCE = 0.01
    }
}
