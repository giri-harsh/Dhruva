package org.anchor.contract

import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * NOTE: written, not run -- see ReplayCsvParserTest for why.
 *
 * Covers VERSIONING.md's compatibility rule directly: MAJOR must match
 * exactly, MINOR/PATCH may be greater-or-equal within that MAJOR. This is
 * the refusal check PRD-ANDROID-ENGINE.md Section2.6 calls "the single most
 * important line of code for the whole compatibility story" -- its own
 * test, not exercised only incidentally through ModelRunner.load.
 */
class ModelManifestTest {

    private fun manifestWithVersion(version: String) = ModelManifest(
        contractVersion = version,
        windowSizeSamples = 20,
        sampleRateHz = 10,
        inputName = "imu_window",
        inputShape = listOf(1, 20, 6),
        featureOrder = listOf("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"),
        normalizationMean = DoubleArray(6),
        normalizationStd = DoubleArray(6) { 1.0 },
        outputMeanName = "velocity_mean_mps",
        outputLogVarianceName = "velocity_log_variance",
        modelSha256 = null,
    )

    @Test
    fun `exact version match is compatible`() {
        assertTrue(manifestWithVersion("1.0.0").isCompatibleWith("1.0.0"))
    }

    @Test
    fun `higher patch within the same major minor is compatible`() {
        assertTrue(manifestWithVersion("1.0.5").isCompatibleWith("1.0.0"))
    }

    @Test
    fun `higher minor within the same major is compatible`() {
        assertTrue(manifestWithVersion("1.2.0").isCompatibleWith("1.0.0"))
    }

    @Test
    fun `lower patch than the minimum is refused`() {
        assertFalse(manifestWithVersion("1.0.0").isCompatibleWith("1.0.5"))
    }

    @Test
    fun `different major is always refused, even if numerically higher`() {
        assertFalse(manifestWithVersion("2.0.0").isCompatibleWith("1.0.0"))
        assertFalse(manifestWithVersion("1.9.9").isCompatibleWith("2.0.0"))
    }

    @Test
    fun `malformed version strings are refused, never treated as compatible by default`() {
        assertFalse(manifestWithVersion("not-a-version").isCompatibleWith("1.0.0"))
        assertFalse(manifestWithVersion("1.0.0").isCompatibleWith("also-not-a-version"))
    }

    @Test
    fun `normalize applies raw minus mean over std per feature, the one place this formula runs`() {
        val manifest = manifestWithVersion("1.0.0").copy(
            normalizationMean = doubleArrayOf(1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
            normalizationStd = doubleArrayOf(2.0, 2.0, 2.0, 1.0, 1.0, 1.0),
        )
        val window = arrayOf(doubleArrayOf(3.0, 4.0, 5.0, 0.1, 0.2, 0.3))

        val result = manifest.normalize(window)

        assertTrue(result[0][0] == 1.0) // (3-1)/2
        assertTrue(result[0][1] == 1.0) // (4-2)/2
        assertTrue(result[0][2] == 1.0) // (5-3)/2
        assertTrue(result[0][3] == 0.1) // (0.1-0)/1
    }
}
