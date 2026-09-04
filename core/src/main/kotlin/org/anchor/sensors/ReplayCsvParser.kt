package org.anchor.sensors

import org.anchor.contract.ReplayCsvSchema
import org.anchor.math.Vec3
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

/** Thrown by ReplayCsvParser. Always carries enough context (file, row,
 *  column where relevant) to fix the file without re-deriving what went
 *  wrong -- the same standard validate_replay_csv.py holds itself to. */
class ReplayCsvValidationException(message: String) : Exception(message)

internal data class ParsedRow(
    val timestampNanos: Long,
    val accel: Vec3,
    val gyro: Vec3,
    val mag: Vec3,
    val gnss: SensorEvent.GnssFix?,
)

/**
 * Pure parse+validate function, deliberately separate from CsvReplaySource's
 * Flow-emitting wrapper: a pure `File -> List<ParsedRow>` (or throws) is
 * trivial to unit-test directly, with no coroutine/Flow test harness
 * needed, and is exactly the "no filter, no UI" first milestone's Stage 1.
 *
 * Enforces every rule in the replay_csv schema's known_failure_modes list,
 * PLUS one rule stricter than the current
 * contracts/replay_csv/validate_replay_csv.py reference: that validator
 * uses pandas' is_monotonic_increasing, which is NON-STRICT and lets
 * duplicate timestamps through despite the schema text itself ("no two
 * rows share a timestamp"). This parser enforces strictly-increasing,
 * matching what the schema actually says. Flagged, not silently fixed --
 * the Python validator is out of scope to edit here.
 */
object ReplayCsvParser {

    internal fun parseAndValidate(path: Path): List<ParsedRow> {
        val raw = Files.readAllBytes(path)

        if (raw.size >= 3 && raw[0] == 0xEF.toByte() && raw[1] == 0xBB.toByte() && raw[2] == 0xBF.toByte()) {
            fail(path, "file has a UTF-8 BOM. Re-save without BOM (encoding must be plain UTF-8).")
        }
        for (b in raw) {
            if (b == '\r'.code.toByte()) {
                fail(
                    path,
                    "file contains a CR byte (CRLF or bare-CR line ending). Must be LF only -- " +
                        "check .gitattributes is applied, or the file was hand-edited on Windows " +
                        "without LF enforcement. A naive line.split(comma) reader would silently " +
                        "include a trailing CR in the last field of every row if this slipped through.",
                )
            }
        }

        val text = String(raw, StandardCharsets.UTF_8)
        val lines = text.split("\n").let { if (it.isNotEmpty() && it.last().isEmpty()) it.dropLast(1) else it }
        if (lines.isEmpty()) fail(path, "file is empty, expected a header row plus data.")

        val header = lines[0].split(",")
        if (header != ReplayCsvSchema.COLUMNS) {
            val missing = ReplayCsvSchema.COLUMNS.toSet() - header.toSet()
            val extra = header.toSet() - ReplayCsvSchema.COLUMNS.toSet()
            val orderNote = if (missing.isEmpty() && extra.isEmpty()) " Columns present but in the WRONG ORDER." else ""
            fail(
                path,
                "header mismatch.$orderNote\n" +
                    "  expected: ${ReplayCsvSchema.COLUMNS}\n" +
                    "  got:      $header\n" +
                    "  missing:  ${missing.sorted()}\n" +
                    "  extra:    ${extra.sorted()}",
            )
        }

        val result = ArrayList<ParsedRow>(lines.size - 1)
        var previousTimestampMs: Long? = null

        for (rowIndex in 1 until lines.size) {
            val line = lines[rowIndex]
            if (line.isEmpty()) continue // tolerate a single trailing blank line, nothing else
            val fields = line.split(",")
            if (fields.size != ReplayCsvSchema.COLUMNS.size) {
                fail(
                    path,
                    "row $rowIndex: expected ${ReplayCsvSchema.COLUMNS.size} columns, got ${fields.size}. " +
                        "If this row has MORE fields than expected, the most likely cause is a locale " +
                        "decimal-separator bug (a comma used as a decimal point, e.g. from Excel on a " +
                        "non-US-locale machine) colliding with the comma delimiter -- never open or " +
                        "re-save this file in Excel. Row content: $line",
                )
            }

            fun numeric(colIndex: Int): Double {
                val colName = ReplayCsvSchema.COLUMNS[colIndex]
                val rawField = fields[colIndex]
                if (rawField.isEmpty()) {
                    if (colName !in ReplayCsvSchema.NULLABLE_COLUMNS) {
                        fail(path, "row $rowIndex, column $colName is empty but this column is not nullable.")
                    }
                    fail(path, "internal: numeric() called on nullable column $colName without a null check first")
                }
                val value = rawField.toDoubleOrNull()
                if (value == null || !value.isFinite()) {
                    fail(
                        path,
                        "row $rowIndex, column $colName value $rawField is not a valid finite number " +
                            "(never NaN, Infinity, the string null, or the string NA -- the schema missing-" +
                            "value convention is an EMPTY field, not any of those tokens).",
                    )
                }
                return value
            }

            val timestampMsRaw = fields[0]
            val timestampMs = timestampMsRaw.toLongOrNull()
                ?: fail(path, "row $rowIndex, column timestamp_ms value $timestampMsRaw is not a valid int64.")
            if (previousTimestampMs != null && timestampMs <= previousTimestampMs) {
                fail(
                    path,
                    "row $rowIndex: timestamp_ms=$timestampMs is not strictly greater than the previous " +
                        "row value $previousTimestampMs. The schema requires that no two rows share a " +
                        "timestamp and that ordering is strictly increasing -- this is checked stricter " +
                        "here than the current Python validate_replay_csv.py, which uses a non-strict " +
                        "monotonic check and would let an exact duplicate through undetected.",
                )
            }
            previousTimestampMs = timestampMs

            val accel = Vec3(numeric(1), numeric(2), numeric(3))
            val gyroRaw = Vec3(numeric(4), numeric(5), numeric(6))
            val mag = Vec3(numeric(7), numeric(8), numeric(9))

            // Enforced HERE, not left to SensorEvent.Gyroscope's constructor further
            // downstream: this parser builds a plain ParsedRow (a Vec3), which
            // CsvReplaySource later wraps into a SensorEvent.Gyroscope -- if the
            // magnitude check lived only in that constructor, a row-numbered
            // diagnostic would be impossible (the constructor has no row context),
            // and calling parseAndValidate() directly, as ReplayCsvParserTest does,
            // would silently accept a deg/s-mistaken-for-rad/s row entirely. Found
            // by actually running the tests, not by inspection -- see the Week-1
            // verification report for how this was caught.
            for ((axisName, value) in listOf("gyro_x" to gyroRaw.x, "gyro_y" to gyroRaw.y, "gyro_z" to gyroRaw.z)) {
                if (kotlin.math.abs(value) >= 10.0) {
                    fail(
                        path,
                        "row $rowIndex, column $axisName has |value| >= 10 rad/s (value=$value, " +
                            "~573 deg/s) -- no road vehicle yaws that fast. This almost certainly " +
                            "means gyro was logged in deg/s, not rad/s (contracts/units.md). " +
                            "Convert with value * pi / 180 at the producer.",
                    )
                }
            }

            val gnssValidRaw = fields[ReplayCsvSchema.GNSS_INDEX_VALID]
            if (gnssValidRaw != "0" && gnssValidRaw != "1") {
                fail(path, "row $rowIndex: gnss_valid must be exactly 0 or 1, got $gnssValidRaw.")
            }
            val gnssValid = gnssValidRaw == "1"

            val gnssFieldIndices = listOf(
                ReplayCsvSchema.GNSS_INDEX_LAT,
                ReplayCsvSchema.GNSS_INDEX_LON,
                ReplayCsvSchema.GNSS_INDEX_SPEED,
                ReplayCsvSchema.GNSS_INDEX_COURSE,
            )
            if (!gnssValid) {
                for (idx in gnssFieldIndices) {
                    if (fields[idx].isNotEmpty()) {
                        fail(
                            path,
                            "row $rowIndex: gnss_valid=0 but column ${ReplayCsvSchema.COLUMNS[idx]} is " +
                                "non-empty (value ${fields[idx]}). GNSS-invalid rows must leave all gnss_* " +
                                "fields empty, not populated -- 0,0 is a real coordinate (off West " +
                                "Africa) and silently corrupts map matching if used as a sentinel.",
                        )
                    }
                }
            } else {
                // Stricter than the Python reference, which does not currently check this
                // direction: gnss_valid=1 asserts a real fix exists, so all four fields
                // are expected present. A row claiming a fix while missing a field is a
                // genuine data-integrity bug, not something to silently half-accept.
                for (idx in gnssFieldIndices) {
                    if (fields[idx].isEmpty()) {
                        fail(
                            path,
                            "row $rowIndex: gnss_valid=1 but column ${ReplayCsvSchema.COLUMNS[idx]} is " +
                                "empty. A row asserting a real GNSS fix must carry all four gnss_* " +
                                "fields -- this check does not exist yet in validate_replay_csv.py, " +
                                "flagged as a gap worth closing there too.",
                        )
                    }
                }
            }

            val gnss = if (gnssValid) {
                SensorEvent.GnssFix(
                    timestampNanos = timestampMs * 1_000_000L,
                    latDeg = numeric(ReplayCsvSchema.GNSS_INDEX_LAT),
                    lonDeg = numeric(ReplayCsvSchema.GNSS_INDEX_LON),
                    speedMps = numeric(ReplayCsvSchema.GNSS_INDEX_SPEED),
                    courseDeg = numeric(ReplayCsvSchema.GNSS_INDEX_COURSE),
                )
            } else null

            // Gyro magnitude was already checked above, with row context. The
            // SensorEvent.Gyroscope constructor this ParsedRow eventually feeds
            // (via CsvReplaySource) carries the SAME assertion independently, as
            // defense-in-depth for any other SensorSource that might construct a
            // Gyroscope directly without going through this parser -- not because
            // this row's check depends on it.
            result.add(
                ParsedRow(
                    timestampNanos = timestampMs * 1_000_000L,
                    accel = accel,
                    gyro = gyroRaw,
                    mag = mag,
                    gnss = gnss,
                ),
            )
        }

        return result
    }

    private fun fail(path: Path, message: String): Nothing =
        throw ReplayCsvValidationException("$path: $message")
}
