package org.anchor.fusion

import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** NOTE: written, not run via Gradle -- see the Week-1 report for why. */
class StationarityDetectorTest {

    private val g = StationarityDetector.GRAVITY_MAGNITUDE

    @Test
    fun `reports not-stationary before the window fills, insufficient history`() {
        val d = StationarityDetector(windowSize = 20)
        repeat(5) { assertFalse(d.addSample(Vec3(0.0, 0.0, g), Vec3.ZERO)) }
    }

    @Test
    fun `a genuinely still phone is detected as stationary once the window fills`() {
        val d = StationarityDetector(windowSize = 20)
        var last = false
        repeat(30) { last = d.addSample(Vec3(0.0, 0.0, g), Vec3(0.0, 0.0, 0.0)) }
        assertTrue(last)
    }

    @Test
    fun `sustained road-vibration-scale accel variance is not classified as stationary`() {
        val d = StationarityDetector(windowSize = 20)
        var last = false
        repeat(30) {
            // A visibly fluctuating accel magnitude -- driving vibration,
            // not sensor noise on an otherwise-still phone.
            val wobble = 1.5 * kotlin.math.sin(it * 1.3)
            last = d.addSample(Vec3(wobble, 0.0, g), Vec3.ZERO)
        }
        assertFalse(last)
    }

    @Test
    fun `sustained yaw motion is not classified as stationary even with a still accelerometer`() {
        val d = StationarityDetector(windowSize = 20)
        var last = false
        repeat(30) {
            val wobble = 0.3 * kotlin.math.sin(it * 0.9)
            last = d.addSample(Vec3(0.0, 0.0, g), Vec3(0.0, 0.0, wobble))
        }
        assertFalse(last)
    }

    @Test
    fun `reset clears history, returning to not-enough-data`() {
        val d = StationarityDetector(windowSize = 20)
        repeat(30) { d.addSample(Vec3(0.0, 0.0, g), Vec3.ZERO) }
        assertTrue(d.isStationary())
        d.reset()
        assertFalse(d.isStationary())
    }
}
