package org.anchor.math

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

/** NOTE: written, not run via Gradle -- see the Week-1 report for why;
 *  verified via direct kotlinc/junit-platform-console-standalone instead
 *  (see this PR own validation section). */
class MatTest {

    private val eps = 1e-9

    @Test
    fun `multiply matches a hand-computed 2x2 case`() {
        // [1 2] [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
        // [3 4] [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
        val a = Mat(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val b = Mat(2, 2, doubleArrayOf(5.0, 6.0, 7.0, 8.0))
        val c = a * b
        assertEquals(19.0, c[0, 0], eps)
        assertEquals(22.0, c[0, 1], eps)
        assertEquals(43.0, c[1, 0], eps)
        assertEquals(50.0, c[1, 1], eps)
    }

    @Test
    fun `matrix times vector matches hand-computed case`() {
        val a = Mat(2, 3, doubleArrayOf(1.0, 0.0, 2.0, 0.0, 1.0, 3.0))
        val v = Vec.of(1.0, 1.0, 1.0)
        val r = a * v
        assertEquals(3.0, r[0], eps) // 1*1 + 0*1 + 2*1
        assertEquals(4.0, r[1], eps) // 0*1 + 1*1 + 3*1
    }

    @Test
    fun `transpose swaps dimensions and elements correctly`() {
        val a = Mat(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val t = a.transpose()
        assertEquals(3, t.rows)
        assertEquals(2, t.cols)
        assertEquals(1.0, t[0, 0], eps)
        assertEquals(4.0, t[0, 1], eps)
        assertEquals(2.0, t[1, 0], eps)
        assertEquals(6.0, t[2, 1], eps)
    }

    @Test
    fun `inverse of a known matrix satisfies M times M-inverse equals identity`() {
        // A well-conditioned, hand-checkable 3x3.
        val a = Mat(3, 3, doubleArrayOf(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0))
        val inv = a.inverse()
        assertEquals(0.5, inv[0, 0], eps)
        assertEquals(1.0 / 3.0, inv[1, 1], eps)
        assertEquals(0.25, inv[2, 2], eps)

        val product = a * inv
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(if (i == j) 1.0 else 0.0, product[i, j], 1e-9)
            }
        }
    }

    @Test
    fun `inverse of a non-diagonal well-conditioned matrix round-trips to identity`() {
        val a = Mat(2, 2, doubleArrayOf(4.0, 7.0, 2.0, 6.0))
        val inv = a.inverse()
        val product = a * inv
        assertEquals(1.0, product[0, 0], 1e-9)
        assertEquals(0.0, product[0, 1], 1e-9)
        assertEquals(0.0, product[1, 0], 1e-9)
        assertEquals(1.0, product[1, 1], 1e-9)
    }

    @Test
    fun `inverse throws on a singular matrix rather than returning garbage`() {
        val singular = Mat(2, 2, doubleArrayOf(1.0, 2.0, 2.0, 4.0)) // row2 = 2*row1
        assertThrows(IllegalStateException::class.java) { singular.inverse() }
    }

    @Test
    fun `identity is a true multiplicative identity`() {
        val a = Mat(3, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        val i = Mat.identity(3)
        val product = a * i
        for (idx in 0 until 9) assertEquals(a.data[idx], product.data[idx], eps)
    }

    @Test
    fun `symmetrized averages off-diagonal asymmetry`() {
        val a = Mat(2, 2, doubleArrayOf(1.0, 3.0, 5.0, 2.0))
        val s = a.symmetrized()
        assertEquals(1.0, s[0, 0], eps)
        assertEquals(4.0, s[0, 1], eps) // (3+5)/2
        assertEquals(4.0, s[1, 0], eps)
        assertEquals(2.0, s[1, 1], eps)
    }

    @Test
    fun `block and setBlock round-trip correctly`() {
        val big = Mat.zeros(5, 5)
        val sub = Mat(2, 2, doubleArrayOf(9.0, 8.0, 7.0, 6.0))
        big.setBlock(1, 2, sub)
        val readBack = big.block(1, 2, 2, 2)
        assertEquals(9.0, readBack[0, 0], eps)
        assertEquals(6.0, readBack[1, 1], eps)
        // Outside the written block must remain zero.
        assertEquals(0.0, big[0, 0], eps)
        assertEquals(0.0, big[4, 4], eps)
    }

    @Test
    fun `skew matrix satisfies skew(v) times w equals v cross w`() {
        val v = Vec3(1.0, 2.0, 3.0)
        val w = Vec3(4.0, 5.0, 6.0)
        val expectedCross = Vec3(
            v.y * w.z - v.z * w.y,
            v.z * w.x - v.x * w.z,
            v.x * w.y - v.y * w.x,
        )
        val skewed = Mat.skew(v) * w
        assertEquals(expectedCross.x, skewed.x, eps)
        assertEquals(expectedCross.y, skewed.y, eps)
        assertEquals(expectedCross.z, skewed.z, eps)
    }

    @Test
    fun `multiply rejects mismatched shapes`() {
        val a = Mat.zeros(2, 3)
        val b = Mat.zeros(2, 3)
        assertThrows(IllegalArgumentException::class.java) { a * b }
    }
}
