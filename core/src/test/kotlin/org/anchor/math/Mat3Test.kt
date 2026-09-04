package org.anchor.math

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

/**
 * Focused test for transpose(), added alongside it in Week 2. The rest of
 * Mat3 is already exercised extensively and correctly by
 * org.anchor.alignment.GravityAlignTest's hand-verified rotation cases --
 * this file does not duplicate that coverage.
 */
class Mat3Test {

    private val eps = 1e-12

    @Test
    fun `transpose swaps off-diagonal elements and preserves diagonal`() {
        val m = Mat3(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        )
        val t = m.transpose()
        assertEquals(1.0, t.m00, eps); assertEquals(4.0, t.m01, eps); assertEquals(7.0, t.m02, eps)
        assertEquals(2.0, t.m10, eps); assertEquals(5.0, t.m11, eps); assertEquals(8.0, t.m12, eps)
        assertEquals(3.0, t.m20, eps); assertEquals(6.0, t.m21, eps); assertEquals(9.0, t.m22, eps)
    }

    @Test
    fun `transpose of identity is identity`() {
        val t = Mat3.IDENTITY.transpose()
        assertEquals(1.0, t.m00, eps); assertEquals(1.0, t.m11, eps); assertEquals(1.0, t.m22, eps)
        assertEquals(0.0, t.m01, eps); assertEquals(0.0, t.m10, eps)
    }

    @Test
    fun `a rotation matrix transpose equals its inverse, R times R-transpose is identity`() {
        // Reuses the exact 90deg-about-plus-x case already hand-derived in
        // GravityAlignTest, as an independent property check here: for any
        // rotation matrix, R * R^T == I.
        val r = Mat3(
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
        )
        val product = r * r.transpose()
        assertEquals(1.0, product.m00, eps); assertEquals(0.0, product.m01, eps); assertEquals(0.0, product.m02, eps)
        assertEquals(0.0, product.m10, eps); assertEquals(1.0, product.m11, eps); assertEquals(0.0, product.m12, eps)
        assertEquals(0.0, product.m20, eps); assertEquals(0.0, product.m21, eps); assertEquals(1.0, product.m22, eps)
    }
}
