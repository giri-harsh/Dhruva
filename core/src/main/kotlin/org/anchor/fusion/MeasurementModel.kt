package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec

/**
 * The generic measurement-update contract every concrete update (NHC,
 * ZUPT, VelocityUpdate, and eventually GNSS) implements. ErrorStateEkf.
 * correct() is written against only this interface -- it has no
 * knowledge of what NHC or ZUPT actually are, which is what "add cleanly"
 * means structurally rather than as an aspiration.
 *
 * predicted() and actual() are kept separate (rather than a single
 * innovation() method) so a caller -- notably the Phase-3E generic-update
 * test -- can assert on each piece independently: predicted measurement,
 * actual measurement, innovation, innovation covariance, gain, and
 * correction are all separately inspectable, not one opaque call.
 */
interface MeasurementModel {
    /** For MODE_EVENT-style logging/diagnostics -- not used in the math. */
    val name: String

    /** Measurement-space dimension. jacobian()/noise() must be
     *  dimension x ErrorStateLayout.DIM and dimension x dimension. */
    val dimension: Int

    /** h(x_nominal): the measurement this model predicts given the
     *  current nominal state. */
    fun predicted(state: NominalState): Vec

    /** z: the actually observed value (captured by the concrete model at
     *  construction time -- e.g. ZUPT/NHC target zero directly; a real
     *  sensor reading for VelocityUpdate/GNSS). */
    fun actual(): Vec

    /** H: the Jacobian of predicted() with respect to the 15-element
     *  error state, evaluated at the current nominal state. */
    fun jacobian(state: NominalState): Mat

    /** R: measurement noise covariance, dimension x dimension. */
    fun noise(): Mat
}
