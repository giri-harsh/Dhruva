package org.anchor.math

/**
 * A plain 3-vector. No allocation-avoidance tricks yet -- that discipline
 * (v3 PRD Sec10.2/Sec9.1: zero allocation in ErrorStateEkf's hot path) matters
 * once the filter is propagating at 100-200 Hz. Alignment/windowing run at
 * <=10 Hz and are not the hot path; premature optimisation here would just
 * be noise. Revisit if EkfZeroAllocationSoakTest (future work) says otherwise.
 */
data class Vec3(val x: Double, val y: Double, val z: Double) {
    operator fun minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
    operator fun plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
    /** Added Week 2 -- org.anchor.fusion needs scalar-times-vector for its
     *  dt-scaled integration terms. Purely additive. */
    operator fun times(scalar: Double) = Vec3(x * scalar, y * scalar, z * scalar)
    fun norm(): Double = kotlin.math.sqrt(x * x + y * y + z * z)

    fun toDoubleArray(): DoubleArray = doubleArrayOf(x, y, z)

    companion object {
        val ZERO = Vec3(0.0, 0.0, 0.0)
        fun of(a: DoubleArray): Vec3 {
            require(a.size == 3) { "Vec3.of expects length 3, got ${a.size}" }
            return Vec3(a[0], a[1], a[2])
        }
    }
}
