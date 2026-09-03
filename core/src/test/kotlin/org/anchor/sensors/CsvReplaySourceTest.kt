package org.anchor.sensors

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

/**
 * NOTE: written, not run -- see ReplayCsvParserTest for why. Correct by
 * inspection against CsvReplaySource own logic.
 */
class CsvReplaySourceTest {

    @TempDir
    lateinit var tempDir: Path

    private val header = ReplayCsvSchema.COLUMNS.joinToString(",")

    private fun writeCsv(name: String, rows: List<String>): Path {
        val path = tempDir.resolve(name)
        val content = (listOf(header) + rows).joinToString("\n") + "\n"
        Files.write(path, content.toByteArray(StandardCharsets.UTF_8))
        return path
    }

    @Test
    fun `emits Accel Gyro Mag GnssFix in that fixed per-row order`() {
        val path = writeCsv(
            "order.csv",
            listOf("1000,0.1,0.2,9.8,0.01,0.02,0.03,20.0,-5.0,40.0,1,28.61,77.20,12.0,45.0"),
        )

        val events = runBlocking { CsvReplaySource(path).events().toList() }

        assertEquals(4, events.size)
        assertTrue(events[0] is SensorEvent.Accelerometer)
        assertTrue(events[1] is SensorEvent.Gyroscope)
        assertTrue(events[2] is SensorEvent.Magnetometer)
        assertTrue(events[3] is SensorEvent.GnssFix)
    }

    @Test
    fun `omits GnssFix entirely for a GNSS-invalid row, no zero-filled placeholder`() {
        val path = writeCsv(
            "nofix.csv",
            listOf("1000,0.1,0.2,9.8,0.01,0.02,0.03,20.0,-5.0,40.0,0,,,,"),
        )

        val events = runBlocking { CsvReplaySource(path).events().toList() }

        assertEquals(3, events.size)
        assertTrue(events.none { it is SensorEvent.GnssFix })
    }

    @Test
    fun `replaying the same file twice yields a byte-identical emitted sequence`() {
        val path = writeCsv(
            "determinism.csv",
            listOf(
                "1000,0.1,0.2,9.8,0.01,0.02,0.03,20.0,-5.0,40.0,1,28.61,77.20,12.0,45.0",
                "1100,0.15,0.22,9.81,0.011,0.021,0.031,20.1,-5.1,40.1,0,,,,",
                "1200,0.2,0.24,9.82,0.012,0.022,0.032,20.2,-5.2,40.2,1,28.62,77.21,12.1,45.1",
            ),
        )

        val firstRun = runBlocking { CsvReplaySource(path).events().toList() }
        val secondRun = runBlocking { CsvReplaySource(path).events().toList() }

        assertEquals(firstRun, secondRun)
        // 3 rows: two with a fix (4 events each) and one without (3 events) = 11
        assertEquals(11, firstRun.size)
    }

    @Test
    fun `clockAnchor reproduces the file own first timestamp as Unix epoch millis`() {
        val path = writeCsv(
            "anchor.csv",
            listOf(
                "1735000000000,0.0,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,",
                "1735000000100,0.0,0.0,9.8,0.0,0.0,0.0,20.0,-5.0,40.0,0,,,,",
            ),
        )
        val source = CsvReplaySource(path)

        val anchor = source.clockAnchor()
        val events = runBlocking { source.events().toList() }
        val secondRowAccel = events.first { it is SensorEvent.Accelerometer && it.timestampNanos > events[0].timestampNanos }

        assertEquals(1735000000000L, anchor.toUnixEpochMillis(events[0].timestampNanos))
        assertEquals(1735000000100L, anchor.toUnixEpochMillis(secondRowAccel.timestampNanos))
    }
}
