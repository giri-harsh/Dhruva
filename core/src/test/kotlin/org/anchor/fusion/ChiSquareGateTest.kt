package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** NOTE: written, not run via Gradle -- see the Week-1 report for why. */
class ChiSquareGateTest {

    @Test
    fun `a small innovation consistent with the covariance is accepted`() {
        val innovation = Vec.of(0.1)
        val s = Mat(1, 1, doubleArrayOf(1.0)) // std=1, a 0.1 deviation is unremarkable
        val result = ChiSquareGate.test(innovation, s)
        assertTrue(result.accepted, "statistic=${result.statistic} threshold=${result.threshold}")
    }

    @Test
    fun `a synthetic jump far outside the innovation covariance is rejected, FR-27 explicit test`() {
        // FR-27's own wording: "a unit test injects a synthetic jump and
        // asserts rejection."
        val innovation = Vec.of(50.0) // a 50-unit jump
        val s = Mat(1, 1, doubleArrayOf(1.0)) // against a std of 1 -- wildly inconsistent
        val result = ChiSquareGate.test(innovation, s)
        assertFalse(result.accepted, "a 50-sigma jump must be rejected")
        assertTrue(result.statistic > result.threshold)
    }

    @Test
    fun `statistic matches the hand-computed Mahalanobis distance for a 2D case`() {
        val innovation = Vec.of(2.0, 1.0)
        val s = Mat.identity(2) * 4.0 // S = 4I -> S^-1 = 0.25*I
        val result = ChiSquareGate.test(innovation, s)
        // nu^T S^-1 nu = 0.25*(4+1) = 1.25
        assertEquals(1.25, result.statistic, 1e-9)
    }

    @Test
    fun `P99 threshold is larger than P95 for the same dimension`() {
        val innovation = Vec.of(3.0)
        val s = Mat(1, 1, doubleArrayOf(1.0))
        val r95 = ChiSquareGate.test(innovation, s, ChiSquareGate.Confidence.P95)
        val r99 = ChiSquareGate.test(innovation, s, ChiSquareGate.Confidence.P99)
        assertTrue(r99.threshold > r95.threshold)
    }

    @Test
    fun `an untabulated dimension fails loudly rather than approximating a threshold`() {
        val innovation = Vec(DoubleArray(10))
        val s = Mat.identity(10)
        assertThrows(IllegalStateException::class.java) { ChiSquareGate.test(innovation, s) }
    }
}
