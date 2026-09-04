package org.anchor.alignment

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * NOTE: written, not run -- see ReplayCsvParserTest for why.
 */
class YawResolverTest {

    @Test
    fun `insufficient samples below 200 falls back to zero, matching the Python gate exactly`() {
        val samples = List(50) { 1.0 to 0.0 }
        val theta = YawResolver.bestYaw(samples) { i -> if (i < 50) 5.0 else null }
        assertEquals(0.0, theta)
    }

    @Test
    fun `reference signal with near-zero variance falls back to zero`() {
        val samples = List(300) { 1.0 to 0.0 }
        // Constant reference (std = 0) -- below the 1e-3 gate.
        val theta = YawResolver.bestYaw(samples) { 2.0 }
        assertEquals(0.0, theta)
    }

    @Test
    fun `recovers a known yaw offset exactly on-grid, with a genuinely unique correlation peak`() {
        // Construction (worked out in the class doc, not asserted blindly): let y be the
        // reference signal and perp an independent, differently-shaped signal. Setting
        //   ax = y*cos(trueTheta) - perp*sin(trueTheta)
        //   ay = y*sin(trueTheta) + perp*cos(trueTheta)
        // makes fwd(theta) = y*cos(theta-trueTheta) + perp*sin(theta-trueTheta) for every
        // candidate theta (a standard angle-subtraction identity) -- which peaks at
        // fwd(trueTheta) = y exactly (correlation 1.0), and is a genuine blend elsewhere,
        // not a degenerate +-1 tie across half the circle the way a pure scalar-multiple
        // construction would be.
        val n = 250
        val trueTheta = PI / 6.0 // 30 degrees -- lands exactly on one of the 72 grid points
        val y = DoubleArray(n) { 5.0 * sin(it * 0.15) }
        val perp = DoubleArray(n) { 3.0 * sin(it * 0.37 + 1.0) }

        val samples = (0 until n).map { i ->
            val ax = y[i] * cos(trueTheta) - perp[i] * sin(trueTheta)
            val ay = y[i] * sin(trueTheta) + perp[i] * cos(trueTheta)
            ax to ay
        }

        val recovered = YawResolver.bestYaw(samples) { i -> y[i] }

        assertEquals(trueTheta, recovered, 1e-6)
    }

    @Test
    fun `yawRotation at zero is identity`() {
        val r = YawResolver.yawRotation(0.0)
        assertEquals(1.0, r.m00, 1e-12); assertEquals(0.0, r.m01, 1e-12)
        assertEquals(0.0, r.m10, 1e-12); assertEquals(1.0, r.m11, 1e-12)
        assertEquals(1.0, r.m22, 1e-12)
    }
}
