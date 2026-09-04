package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec

/**
 * FR-11: fuses ANCHOR-Net's predicted forward speed, weighted by its own
 * predicted variance. mean/variance are the model's own output units --
 * confirmed by Harshit in contracts/VERSIONING.md's "Resolved cross-track
 * questions" section: velocity_mean_mps is mean forward SPEED in m/s, not
 * displacement, so this measures the vehicle-frame FORWARD (x) velocity
 * component directly, no per-window-duration conversion needed.
 *
 * variance MUST already be exp(velocity_log_variance), per
 * contracts/model_io/model_manifest.json's compatibility_rule -- this
 * class takes the already-converted variance, it does not do that
 * conversion itself (ModelRunner's own doc already assigns that
 * conversion to exactly one place: its own output-parsing code).
 *
 * Shares BodyVelocityJacobian with NhcUpdate (see that file) -- this is
 * the SAME body-velocity observation, just the forward (x) row instead
 * of the lateral/vertical (y,z) rows.
 *
 * FR-11's own acceptance criterion ("doubling sigma^2 halves the state
 * correction magnitude") holds exactly when R dominates the innovation
 * covariance S = HPH^T + R -- i.e. when the filter's own uncertainty in
 * forward velocity is small relative to the model's stated noise, which
 * is the realistic, meaningful regime this property is meant to hold in
 * (if HPH^T dominated instead, the filter would nearly fully trust any
 * measurement regardless of its claimed noise, which is not what FR-11
 * wants). VelocityUpdateTest constructs its covariance in that regime
 * deliberately, not by coincidence -- see that test's own comment.
 */
class VelocityUpdate(
    private val meanMps: Double,
    private val variance: Double,
    private val trustFactor: Double = 1.0,
) : MeasurementModel {
    init {
        require(variance > 0.0) { "VelocityUpdate variance must be positive, got $variance" }
        require(trustFactor > 0.0) { "VelocityUpdate trustFactor must be positive, got $trustFactor" }
    }

    override val name: String = "VelocityUpdate"
    override val dimension: Int = 1

    override fun predicted(state: NominalState): Vec = Vec.of(state.bodyVelocity().x)

    override fun actual(): Vec = Vec.of(meanMps)

    override fun jacobian(state: NominalState): Mat = BodyVelocityJacobian.selectRows(state, intArrayOf(0))

    /** R = sigma^2 * trust_factor, exactly FR-11's formula. */
    override fun noise(): Mat = Mat(1, 1, doubleArrayOf(variance * trustFactor))
}
