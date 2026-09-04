package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec

/**
 * FR-10: non-holonomic constraint. A ground vehicle does not slide
 * sideways or jump vertically, so body-frame lateral (y) and vertical (z)
 * velocity are observed as zero -- vehicle frame per
 * contracts/frame_convention.md (x=forward, y=left, z=up).
 *
 * Jacobian: rows y,z of BodyVelocityJacobian.full3x15 -- see that file's
 * doc for the derivation; NHC and VelocityUpdate share it, they only
 * select different rows.
 *
 * FR-10 also requires NHC be suppressed when stationary (FR-26, ZUPT
 * takes over) or reversing (sign of forward body velocity, once a gear
 * signal or its proxy exists) -- that gating is an orchestrator-level
 * concern (which update to call this tick), not something NhcUpdate
 * itself decides; this class always produces a valid update when asked,
 * exactly as MeasurementModel's contract promises.
 */
class NhcUpdate(
    private val lateralVerticalNoiseVariance: Double = DEFAULT_NOISE_VARIANCE,
) : MeasurementModel {
    override val name: String = "NHC"
    override val dimension: Int = 2

    override fun predicted(state: NominalState): Vec {
        val bodyVel = state.bodyVelocity()
        return Vec.of(bodyVel.y, bodyVel.z)
    }

    override fun actual(): Vec = Vec.of(0.0, 0.0)

    override fun jacobian(state: NominalState): Mat = BodyVelocityJacobian.selectRows(state, intArrayOf(1, 2))

    override fun noise(): Mat = Mat.diagonal(doubleArrayOf(lateralVerticalNoiseVariance, lateralVerticalNoiseVariance))

    companion object {
        /** (m/s)^2 -- a fixed default per FR-10 ("covariance from the
         *  context head or a fixed default"); Head C (the context head)
         *  is Harshit's, not yet available to adapt this per FR-10's
         *  other named option. [VERIFY] against real vehicle dynamics. */
        const val DEFAULT_NOISE_VARIANCE = 0.05
    }
}
