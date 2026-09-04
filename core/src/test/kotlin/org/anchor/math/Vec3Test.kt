package org.anchor.math

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

/** Focused test for times(Double), added alongside it in Week 2. */
class Vec3Test {
    @Test
    fun `scalar multiply scales every component`() {
        val v = Vec3(1.0, -2.0, 3.0) * 2.5
        assertEquals(2.5, v.x, 1e-12)
        assertEquals(-5.0, v.y, 1e-12)
        assertEquals(7.5, v.z, 1e-12)
    }

    @Test
    fun `scalar multiply by zero gives the zero vector`() {
        val v = Vec3(5.0, -5.0, 5.0) * 0.0
        assertEquals(0.0, v.x, 1e-12); assertEquals(0.0, v.y, 1e-12); assertEquals(0.0, v.z, 1e-12)
    }
}
