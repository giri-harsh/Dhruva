package org.anchor.math

import kotlin.math.sqrt

/**
 * General-dimension vector, DoubleArray-backed. Distinct from Vec3
 * (alignment package own fixed-3D type, unchanged, still used exactly as
 * before) -- the ESKF error state is 15-wide and the nominal state is
 * 16-wide (quaternion), neither of which fits a fixed-3 type. Vec3
 * interop is via toVec()/writeInto() below rather than merging the two
 * types, so Week 1 alignment code needs zero changes.
 *
 * This is the readable, allocating variant -- correct and testable first,
 * per this week own explicit instruction. The zero-allocation hot-path
 * pass (PRD Section5.3, Week 3 by the PRD own phasing) works from this
 * same API surface later; it is not done here.
 */
class Vec(val values: DoubleArray) {
    val size: Int get() = values.size

    operator fun get(i: Int): Double = values[i]
    operator fun set(i: Int, v: Double) { values[i] = v }

    operator fun plus(other: Vec): Vec {
        require(size == other.size) { "Vec size mismatch: $size vs ${other.size}" }
        return Vec(DoubleArray(size) { values[it] + other.values[it] })
    }

    operator fun minus(other: Vec): Vec {
        require(size == other.size) { "Vec size mismatch: $size vs ${other.size}" }
        return Vec(DoubleArray(size) { values[it] - other.values[it] })
    }

    operator fun times(scalar: Double): Vec = Vec(DoubleArray(size) { values[it] * scalar })

    fun dot(other: Vec): Double {
        require(size == other.size) { "Vec size mismatch: $size vs ${other.size}" }
        var s = 0.0
        for (i in values.indices) s += values[i] * other.values[i]
        return s
    }

    fun norm(): Double = sqrt(dot(this))

    /** Read a 3-element slice starting at [offset] as a Vec3, for interop
     *  with alignment/other fixed-3D code. */
    fun slice3(offset: Int): Vec3 {
        require(offset + 3 <= size) { "slice3 offset $offset out of bounds for size $size" }
        return Vec3(values[offset], values[offset + 1], values[offset + 2])
    }

    /** Write a Vec3 into a 3-element slice starting at [offset], in place. */
    fun writeSlice3(offset: Int, v: Vec3) {
        require(offset + 3 <= size) { "writeSlice3 offset $offset out of bounds for size $size" }
        values[offset] = v.x
        values[offset + 1] = v.y
        values[offset + 2] = v.z
    }

    /** As a single-column Mat, for matrix algebra (e.g. P*H^T where H is a Mat). */
    fun toColumnMat(): Mat {
        val m = Mat.zeros(size, 1)
        for (i in 0 until size) m[i, 0] = values[i]
        return m
    }

    fun copy(): Vec = Vec(values.copyOf())

    override fun toString(): String = values.joinToString(prefix = "[", postfix = "]") { "%.6g".format(it) }

    companion object {
        fun zeros(n: Int) = Vec(DoubleArray(n))
        fun of(vararg xs: Double) = Vec(xs)
    }
}
