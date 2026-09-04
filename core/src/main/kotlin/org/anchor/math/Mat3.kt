package org.anchor.math

/**
 * Row-major 3x3 matrix, `m[row][col]`. This is the rotation-matrix type used
 * throughout alignment (org.anchor.alignment) -- it is deliberately the
 * on-device equivalent of the plain NumPy 3x3 arrays `ml/anchor/data/
 * features.py` operates on, not a general-purpose linear-algebra library.
 *
 * Convention, matching features.py exactly (see AlignmentTest for the
 * cross-check): for a rotation matrix R and a device-frame vector v,
 * `R.times(v)` gives the vehicle-frame vector -- i.e. v is a COLUMN vector,
 * R acts on the left. features.py's `lin @ R.T` (a batch of ROW vectors
 * against R transposed) is the numpy idiom for exactly the same operation
 * applied per-row; Mat3.times(Vec3) here is the natural single-sample form
 * of that same math, not a different convention.
 */
data class Mat3(
    val m00: Double, val m01: Double, val m02: Double,
    val m10: Double, val m11: Double, val m12: Double,
    val m20: Double, val m21: Double, val m22: Double,
) {
    operator fun times(v: Vec3): Vec3 = Vec3(
        x = m00 * v.x + m01 * v.y + m02 * v.z,
        y = m10 * v.x + m11 * v.y + m12 * v.z,
        z = m20 * v.x + m21 * v.y + m22 * v.z,
    )

    operator fun times(other: Mat3): Mat3 = Mat3(
        m00 = m00 * other.m00 + m01 * other.m10 + m02 * other.m20,
        m01 = m00 * other.m01 + m01 * other.m11 + m02 * other.m21,
        m02 = m00 * other.m02 + m01 * other.m12 + m02 * other.m22,
        m10 = m10 * other.m00 + m11 * other.m10 + m12 * other.m20,
        m11 = m10 * other.m01 + m11 * other.m11 + m12 * other.m21,
        m12 = m10 * other.m02 + m11 * other.m12 + m12 * other.m22,
        m20 = m20 * other.m00 + m21 * other.m10 + m22 * other.m20,
        m21 = m20 * other.m01 + m21 * other.m11 + m22 * other.m21,
        m22 = m20 * other.m02 + m21 * other.m12 + m22 * other.m22,
    )

    operator fun plus(other: Mat3): Mat3 = Mat3(
        m00 + other.m00, m01 + other.m01, m02 + other.m02,
        m10 + other.m10, m11 + other.m11, m12 + other.m12,
        m20 + other.m20, m21 + other.m21, m22 + other.m22,
    )

    operator fun times(scalar: Double): Mat3 = Mat3(
        m00 * scalar, m01 * scalar, m02 * scalar,
        m10 * scalar, m11 * scalar, m12 * scalar,
        m20 * scalar, m21 * scalar, m22 * scalar,
    )

    /** Added Week 2 (org.anchor.fusion needs R^T for body<->nav velocity
     *  Jacobians) -- purely additive, does not change any existing
     *  Mat3 behaviour or the alignment code that already uses this class. */
    fun transpose(): Mat3 = Mat3(
        m00, m10, m20,
        m01, m11, m21,
        m02, m12, m22,
    )

    companion object {
        val IDENTITY = Mat3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )

        /** diag(a, b, c) */
        fun diagonal(a: Double, b: Double, c: Double) = Mat3(
            a, 0.0, 0.0,
            0.0, b, 0.0,
            0.0, 0.0, c,
        )
    }
}
