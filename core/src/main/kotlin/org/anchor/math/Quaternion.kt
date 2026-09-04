package org.anchor.math

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Unit quaternion, Hamilton convention, SCALAR-FIRST storage (w, x, y, z)
 * -- matching Android own SensorManager.getQuaternionFromVector output
 * order, chosen deliberately since this is a Kotlin/Android-target
 * codebase. Represents a body-to-nav rotation: rotate(q, v_body) ==
 * v_nav, the SAME directional convention Mat3 already uses in
 * org.anchor.alignment (GravityAlign/YawResolver: "R.times(v) gives the
 * vehicle-frame vector") -- extended here to quaternions, not a new
 * convention invented for the filter.
 */
data class Quaternion(val w: Double, val x: Double, val y: Double, val z: Double) {

    /** Hamilton product: this ⊗ other. Composes rotations left-to-right
     *  as (this ⊗ other) applied to a vector means "rotate by other
     *  first, then by this" -- standard Hamilton composition order. */
    operator fun times(o: Quaternion): Quaternion = Quaternion(
        w = w * o.w - x * o.x - y * o.y - z * o.z,
        x = w * o.x + x * o.w + y * o.z - z * o.y,
        y = w * o.y - x * o.z + y * o.w + z * o.x,
        z = w * o.z + x * o.y - y * o.x + z * o.w,
    )

    fun norm(): Double = sqrt(w * w + x * x + y * y + z * z)

    fun normalized(): Quaternion {
        val n = norm()
        check(n > 1e-12) { "Quaternion.normalized(): near-zero norm, cannot normalize" }
        return Quaternion(w / n, x / n, y / n, z / n)
    }

    /** Conjugate == inverse for a unit quaternion; represents nav-to-body. */
    fun conjugate(): Quaternion = Quaternion(w, -x, -y, -z)

    /**
     * Body-to-nav rotation matrix. Active rotation, right-handed: for a
     * body-frame vector v, this.toRotationMatrix() * v == the same
     * physical vector expressed in nav frame. Standard Hamilton
     * quaternion-to-matrix formula (e.g. Sola 2017 eq.62).
     */
    fun toRotationMatrix(): Mat3 {
        val ww = w * w; val xx = x * x; val yy = y * y; val zz = z * z
        val wx = w * x; val wy = w * y; val wz = w * z
        val xy = x * y; val xz = x * z; val yz = y * z
        return Mat3(
            ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz,
        )
    }

    /** Rotates a body-frame vector into nav frame: R(this) * v. */
    fun rotate(v: Vec3): Vec3 = toRotationMatrix() * v

    companion object {
        val IDENTITY = Quaternion(1.0, 0.0, 0.0, 0.0)

        /**
         * Exponential map: a body-frame rotation vector phi (axis * angle,
         * radians) to the unit quaternion representing that rotation.
         * Used both for integrating gyro measurements during propagation
         * (phi = (omega_m - b_g) * dt) and for injecting a delta-theta
         * error-state correction into the nominal quaternion after a
         * measurement update -- same function, same math, two callers.
         *
         * Small-angle branch avoids a 0/0 in sin(theta/2)/theta as
         * theta -> 0; first-order series sin(x)/x ~ 1 - x^2/6 is exact
         * enough for the sub-degree per-step rotations propagate() sees
         * at any realistic IMU rate.
         */
        fun fromRotationVector(phi: Vec3): Quaternion {
            val theta = phi.norm()
            return if (theta < 1e-8) {
                Quaternion(1.0, phi.x / 2.0, phi.y / 2.0, phi.z / 2.0).normalized()
            } else {
                val half = theta / 2.0
                val s = sin(half) / theta
                Quaternion(cos(half), phi.x * s, phi.y * s, phi.z * s)
            }
        }
    }
}
