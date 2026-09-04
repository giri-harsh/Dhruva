package org.anchor.sensors

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.nio.file.Path

/**
 * SensorSource implementation backed by a contracts/replay_csv-schema file.
 * This IS the demo mechanism (PRD-ANDROID-ENGINE.md S-06, v3 PRD Section19.2:
 * "the single most load-bearing component in the demo") and the first
 * working milestone target: same file replayed twice must emit a byte-
 * identical event sequence.
 *
 * Determinism is structural here, not incidental: parseAndValidate reads
 * the file top-to-bottom once into an ordered, immutable list before
 * anything is emitted -- there is no concurrency, no interleaving of
 * per-column streams, and no reordering that could vary between runs.
 * Within one row, events are always emitted in the same fixed order:
 * Accelerometer, then Gyroscope, then Magnetometer, then GnssFix if
 * present. That fixed sub-order is itself part of what "deterministic"
 * means here -- change it and every downstream consumer that assumes a
 * stable per-tick event order breaks silently.
 *
 * The schema carries no gravity_x/y/z column (see
 * contracts/frame_convention.md and the Week-1 report this session
 * produced) -- CsvReplaySource therefore never emits a
 * SensorEvent.Gravity directly. Gravity is derived downstream from the
 * Accelerometer stream, by org.anchor.prefilter.GravityLowPassEstimator,
 * identically for both this source and AndroidSensorSource when a live
 * TYPE_GRAVITY reading is not being used. That derivation is deliberately
 * NOT done here -- CsvReplaySource's only job is "emit exactly what this
 * file says, validated," nothing more.
 */
class CsvReplaySource(private val path: Path) : SensorSource {

    override fun events(): Flow<SensorEvent> = flow {
        val rows = ReplayCsvParser.parseAndValidate(path)
        for (row in rows) {
            emit(SensorEvent.Accelerometer(row.timestampNanos, row.accel))
            emit(SensorEvent.Gyroscope(row.timestampNanos, row.gyro))
            emit(SensorEvent.Magnetometer(row.timestampNanos, row.mag))
            row.gnss?.let { emit(it) }
        }
    }

    /**
     * A ClockAnchor consistent with this file: the first row own
     * timestamp_ms IS the wall-clock anchor, and that row synthesised
     * timestampNanos is the monotonic anchor -- so converting any later
     * event back to Unix-epoch millis reproduces exactly what the file
     * declared, not whatever wall-clock time the replay happened to run
     * at. Exposed so TripExporter (future work) never has to invent its
     * own anchor for a replay session.
     */
    fun clockAnchor(): ClockAnchor {
        val firstRow = ReplayCsvParser.parseAndValidate(path).firstOrNull()
            ?: throw ReplayCsvValidationException("$path: no data rows, cannot anchor a clock")
        val firstTimestampMs = firstRow.timestampNanos / 1_000_000L
        return ClockAnchor.capturedNow(
            monotonicNanosNow = firstRow.timestampNanos,
            wallClockMillisNow = firstTimestampMs,
        )
    }
}
