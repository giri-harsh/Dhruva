package org.anchor.sensors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

/**
 * One negative-case test per rule ReplayCsvParser enforces -- mirroring
 * how the task instructions describe validate_replay_csv.py own test
 * coverage should read: "one malformed fixture per schema rule, each
 * asserting a loud, specific failure, never a silent pass."
 *
 * NOTE: written, not run. This dev environment has no JDK 17+/Gradle
 * (see the Week-1 report) -- these tests are correct by inspection against
 * ReplayCsvParser own logic and against contracts/replay_csv/schema.json,
 * but have not been executed. Run with: ./gradlew :core:test --tests
 * "org.anchor.sensors.ReplayCsvParserTest"
 */
class ReplayCsvParserTest {

    @TempDir
    lateinit var tempDir: Path

    private fun writeCsv(name: String, content: String): Path {
        val path = tempDir.resolve(name)
        Files.write(path, content.toByteArray(StandardCharsets.UTF_8))
        return path
    }

    private val header = ReplayCsvSchema.COLUMNS.joinToString(",")

    @Test
    fun `valid two-row file with one GNSS-valid and one GNSS-invalid row parses correctly`() {
        val csv = listOf(
            header,
            "1000,0.1,0.2,9.8,0.01,0.02,0.03,20.0,-5.0,40.0,1,28.61,77.20,12.0,45.0",
            "1100,0.15,0.22,9.81,0.011,0.021,0.031,20.1,-5.1,40.1,0,,,,",
        ).joinToString("\n") + "\n"
        val path = writeCsv("valid.csv", csv)

        val rows = ReplayCsvParser.parseAndValidate(path)

        assertEquals(2, rows.size)
        assertEquals(1000L * 1_000_000L, rows[0].timestampNanos)
        assertEquals(0.1, rows[0].accel.x)
        assertTrue(rows[0].gnss != null)
        assertEquals(28.61, rows[0].gnss!!.latDeg)
        assertTrue(rows[1].gnss == null)
    }

    @Test
    fun `rejects a UTF-8 BOM`() {
        val path = tempDir.resolve("bom.csv")
        val bom = byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte())
        val body = (header + "\n").toByteArray(StandardCharsets.UTF_8)
        Files.write(path, bom + body)

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("BOM"), ex.message)
    }

    @Test
    fun `rejects CRLF line endings`() {
        val csv = header + "\r\n" +
            "1000,0.0,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,\r\n"
        val path = writeCsv("crlf.csv", csv)

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("CR byte"), ex.message)
    }

    @Test
    fun `rejects a header with columns out of order`() {
        val scrambled = ReplayCsvSchema.COLUMNS.reversed().joinToString(",")
        val path = writeCsv("scrambled.csv", scrambled + "\n")

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("WRONG ORDER"), ex.message)
    }

    @Test
    fun `rejects a row with too many fields, hinting at locale decimal separator`() {
        val csv = header + "\n" +
            "1000,0,0,9,8,0,0,0,20,-5,40,0,,,,\n" // accel_z "9,8" split by the locale bug
        val path = writeCsv("locale.csv", csv)

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("locale decimal-separator"), ex.message)
    }

    @Test
    fun `rejects a non-numeric field`() {
        val csv = header + "\n" +
            "1000,abc,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,\n"
        val path = writeCsv("nonnumeric.csv", csv)

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("not a valid finite number"), ex.message)
    }

    @Test
    fun `rejects an empty non-nullable field`() {
        val csv = header + "\n" +
            "1000,,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,\n"
        val path = writeCsv("emptynonnull.csv", csv)

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("not nullable"), ex.message)
    }

    @Test
    fun `rejects gnss_valid 0 with a populated gnss field, zero is a real coordinate`() {
        val csv = header + "\n" +
            "1000,0.0,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,0.0,0.0,,\n"
        val path = writeCsv("zerosentinel.csv", csv)

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("gnss_valid=0"), ex.message)
    }

    @Test
    fun `rejects gnss_valid 1 with a missing gnss field, stricter than the Python reference`() {
        val csv = header + "\n" +
            "1000,0.0,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,1,28.6,,12.0,45.0\n"
        val path = writeCsv("missingfield.csv", csv)

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("gnss_valid=1"), ex.message)
    }

    @Test
    fun `rejects a duplicate timestamp, stricter than the Python reference own non-strict check`() {
        val csv = listOf(
            header,
            "1000,0.0,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,",
            "1000,0.1,0.1,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,",
        ).joinToString("\n") + "\n"
        val path = writeCsv("duplicatets.csv", csv)

        val ex = assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("strictly greater"), ex.message)
    }

    @Test
    fun `rejects a decreasing timestamp`() {
        val csv = listOf(
            header,
            "2000,0.0,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,",
            "1000,0.1,0.1,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,",
        ).joinToString("\n") + "\n"
        val path = writeCsv("decreasingts.csv", csv)

        assertThrows(ReplayCsvValidationException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
    }

    @Test
    fun `rejects gyro magnitude at or above 10 rad per second, the deg-vs-rad bug`() {
        // 15 deg/s * pi/180 is well under 10 rad/s -- 15.0 as a raw value is the actual
        // deg/s-mistaken-for-rad/s failure mode this exists to catch.
        val csv = header + "\n" +
            "1000,0.0,0.0,9.8,15.0,0.0,0.0,20.0,-5.0,40.0,0,,,,\n"
        val path = writeCsv("badgyro.csv", csv)

        val ex = assertThrows(IllegalArgumentException::class.java) {
            ReplayCsvParser.parseAndValidate(path)
        }
        assertTrue(ex.message!!.contains("deg/s-vs-rad/s"), ex.message)
    }

    @Test
    fun `parsing the same valid file twice yields an identical result, the determinism requirement`() {
        val csv = listOf(
            header,
            "1000,0.1,0.2,9.8,0.01,0.02,0.03,20.0,-5.0,40.0,1,28.61,77.20,12.0,45.0",
            "1100,0.15,0.22,9.81,0.011,0.021,0.031,20.1,-5.1,40.1,0,,,,",
            "1200,0.2,0.24,9.82,0.012,0.022,0.032,20.2,-5.2,40.2,1,28.62,77.21,12.1,45.1",
        ).joinToString("\n") + "\n"
        val path = writeCsv("determinism.csv", csv)

        val firstRun = ReplayCsvParser.parseAndValidate(path)
        val secondRun = ReplayCsvParser.parseAndValidate(path)

        assertEquals(firstRun, secondRun)
        assertFalse(firstRun.isEmpty())
    }
}
