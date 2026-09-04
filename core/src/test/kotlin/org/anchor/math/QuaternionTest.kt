package org.anchor.math

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** NOTE: written, not run via Gradle -- see the Week-1 report for why. */
class QuaternionTest {

    private val eps = 1e-9

    @Test
    fun `identity quaternion leaves a vector unchanged`() {
        val v = Vec3(3.0, -2.0, 5.0)
        val r = Quaternion.IDENTITY.rotate(v)
        assertEquals(v.x, r.x, eps); assertEquals(v.y, r.y, eps); assertEquals(v.z, r.z, eps)
    }

    @Test
    fun `90 degree rotation about z maps plus-x to plus-y, right-hand rule`() {
        // Hand-constructed: 90deg about z is (cos45, 0, 0, sin45).
        val half = PI / 4.0
        val q = Quaternion(cos(half), 0.0, 0.0, sin(half))
        val r = q.rotate(Vec3(1.0, 0.0, 0.0))
        assertEquals(0.0, r.x, 1e-9)
        assertEquals(1.0, r.y, 1e-9)
        assertEquals(0.0, r.z, 1e-9)
    }

    @Test
    fun `fromRotationVector at pi over 2 about z matches the hand-constructed quaternion`() {
        val q = Quaternion.fromRotationVector(Vec3(0.0, 0.0, PI / 2.0))
        val half = PI / 4.0
        assertEquals(cos(half), q.w, 1e-9)
        assertEquals(0.0, q.x, 1e-9)
        assertEquals(0.0, q.y, 1e-9)
        assertEquals(sin(half), q.z, 1e-9)
    }

    @Test
    fun `fromRotationVector small-angle branch does not throw and stays near identity`() {
        val q = Quaternion.fromRotationVector(Vec3(1e-10, 0.0, 0.0))
        assertEquals(1.0, q.w, 1e-6)
        assertEquals(1.0, q.norm(), 1e-9)
    }

    @Test
    fun `fromRotationVector zero vector gives identity`() {
        val q = Quaternion.fromRotationVector(Vec3(0.0, 0.0, 0.0))
        assertEquals(1.0, q.w, eps); assertEquals(0.0, q.x, eps)
        assertEquals(0.0, q.y, eps); assertEquals(0.0, q.z, eps)
    }

    @Test
    fun `composed rotation matches sequential rotation, Hamilton product order`() {
        // (q1 * q2).rotate(v) must equal q1.rotate(q2.rotate(v)) -- a genuine
        // cross-check of composition order, not the same code path twice.
        val q1 = Quaternion.fromRotationVector(Vec3(0.0, 0.0, PI / 3.0))
        val q2 = Quaternion.fromRotationVector(Vec3(0.0, PI / 6.0, 0.0))
        val v = Vec3(1.0, 0.5, -0.3)

        val composed = (q1 * q2).rotate(v)
        val sequential = q1.rotate(q2.rotate(v))

        assertEquals(sequential.x, composed.x, 1e-9)
        assertEquals(sequential.y, composed.y, 1e-9)
        assertEquals(sequential.z, composed.z, 1e-9)
    }

    @Test
    fun `normalized returns unit norm from a scaled quaternion`() {
        val q = Quaternion(2.0, 0.0, 0.0, 0.0).normalized()
        assertEquals(1.0, q.norm(), eps)
        assertEquals(1.0, q.w, eps)
    }

    @Test
    fun `toRotationMatrix of identity is the identity matrix`() {
        val m = Quaternion.IDENTITY.toRotationMatrix()
        assertEquals(1.0, m.m00, eps); assertEquals(0.0, m.m01, eps); assertEquals(0.0, m.m02, eps)
        assertEquals(0.0, m.m10, eps); assertEquals(1.0, m.m11, eps); assertEquals(0.0, m.m12, eps)
        assertEquals(0.0, m.m20, eps); assertEquals(0.0, m.m21, eps); assertEquals(1.0, m.m22, eps)
    }

    @Test
    fun `conjugate inverts a rotation, round trip returns the original vector`() {
        val q = Quaternion.fromRotationVector(Vec3(0.2, -0.4, 0.1))
        val v = Vec3(1.0, 2.0, 3.0)
        val roundTrip = q.conjugate().rotate(q.rotate(v))
        assertEquals(v.x, roundTrip.x, 1e-9)
        assertEquals(v.y, roundTrip.y, 1e-9)
        assertEquals(v.z, roundTrip.z, 1e-9)
    }
}
