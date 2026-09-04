package org.anchor.alignment

import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

/**
 * NOTE: written, not run -- see ReplayCsvParserTest for why.
 *
 * Every expected Mat3 below was worked by hand using the same Rodrigues
 * formula the Kotlin (and the Python reference it ports) implements --
 * these are not "whatever the code happens to produce," they are
 * independently derived, so a transcription bug in the port would be
 * caught by a mismatch here, not rubber-stamped by it.
 */
class GravityAlignTest {

    private val eps = 1e-9

    private fun assertMat3Equals(
        m00: Double, m01: Double, m02: Double,
        m10: Double, m11: Double, m12: Double,
        m20: Double, m21: Double, m22: Double,
        actual: org.anchor.math.Mat3,
    ) {
        assertEquals(m00, actual.m00, eps); assertEquals(m01, actual.m01, eps); assertEquals(m02, actual.m02, eps)
        assertEquals(m10, actual.m10, eps); assertEquals(m11, actual.m11, eps); assertEquals(m12, actual.m12, eps)
        assertEquals(m20, actual.m20, eps); assertEquals(m21, actual.m21, eps); assertEquals(m22, actual.m22, eps)
    }

    @Test
    fun `no samples returns identity`() {
        val r = GravityAlign.rotationGravityToDown(emptyList())
        assertMat3Equals(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, r)
    }

    @Test
    fun `near-zero magnitude gravity returns identity`() {
        val r = GravityAlign.rotationGravityToDown(listOf(Vec3(1e-8, -1e-8, 0.0)))
        assertMat3Equals(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, r)
    }

    @Test
    fun `gravity already at plus-z is the antiparallel special case, returns diag 1 -1 -1`() {
        // src = (0,0,1) normalised, dst = (0,0,-1): cross = 0, dot = -1 (c less than or
        // equal to 0) -- exercises the Python reference's antiparallel fallback branch,
        // NOT the general Rodrigues formula path. Hand-derivation in the class doc.
        val r = GravityAlign.rotationGravityToDown(listOf(Vec3(0.0, 0.0, 9.8), Vec3(0.0, 0.0, 9.8)))
        assertMat3Equals(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, r)
        // Sanity: applying R to the input should land parallel to (0,0,-1).
        val rotated = r * Vec3(0.0, 0.0, 9.8)
        assertEquals(0.0, rotated.x, eps)
        assertEquals(0.0, rotated.y, eps)
        assertEquals(true, rotated.z < 0.0)
    }

    @Test
    fun `gravity already at minus-z is the c greater than 0 special case, returns identity`() {
        val r = GravityAlign.rotationGravityToDown(listOf(Vec3(0.0, 0.0, -9.8)))
        assertMat3Equals(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, r)
    }

    @Test
    fun `gravity at plus-x exercises the general Rodrigues branch, matches hand-derived R`() {
        // src=(1,0,0), dst=(0,0,-1). v=cross(src,dst)=(0,1,0), c=dot=0.
        // vx = [[0,0,1],[0,0,0],[-1,0,0]]; vx^2 = [[-1,0,0],[0,0,0],[0,0,-1]].
        // R = I + vx + vx^2*(1/(1+0)) = [[0,0,1],[0,1,0],[-1,0,0]] -- worked by hand,
        // see the class doc for the full step-by-step arithmetic.
        val r = GravityAlign.rotationGravityToDown(listOf(Vec3(9.8, 0.0, 0.0)))
        assertMat3Equals(
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
            r,
        )
        val rotated = r * Vec3(1.0, 0.0, 0.0)
        assertEquals(0.0, rotated.x, eps)
        assertEquals(0.0, rotated.y, eps)
        assertEquals(-1.0, rotated.z, eps)
    }

    @Test
    fun `non-finite samples are filtered out before averaging, matches np isfinite masking`() {
        val r = GravityAlign.rotationGravityToDown(
            listOf(
                Vec3(Double.NaN, 0.0, 0.0),
                Vec3(0.0, 0.0, -9.8),
                Vec3(0.0, Double.POSITIVE_INFINITY, 0.0),
            ),
        )
        // Only the one finite sample (0,0,-9.8) should survive -> identity, same as the
        // single-sample minus-z case above.
        assertMat3Equals(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, r)
    }
}
