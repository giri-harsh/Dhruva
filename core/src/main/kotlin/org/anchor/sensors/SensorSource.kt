package org.anchor.sensors

import kotlinx.coroutines.flow.Flow

/**
 * The one interface the whole engine consumes sensor data through. A phone
 * (AndroidSensorSource, not yet built this week), a CSV replay
 * (CsvReplaySource), and a 200 Hz serial IMU (SerialImuSource, FR-21, not
 * yet built) are three implementations of this and nothing else --
 * FR-18's "replay through the identical code path as live sensors" is true
 * because of this interface's existence, not by convention.
 *
 * `events()` MUST emit in strictly non-decreasing timestampNanos order.
 * For CsvReplaySource that is a direct, deterministic consequence of
 * reading rows top-to-bottom from a validated file -- see
 * CsvReplaySourceTest's determinism assertion (same file, two runs, byte-
 * identical emitted sequence).
 */
interface SensorSource {
    fun events(): Flow<SensorEvent>
}

/**
 * Converts a source's own monotonic timestampNanos into a genuine Unix-
 * epoch millisecond value, at exactly one point (FR-33: "you must convert
 * deliberately... anchor it against a wall-clock reference sample taken
 * once at stream start"). Nothing else in the engine may do this
 * conversion inline -- if you find yourself writing `timestampNanos / 1e6`
 * anywhere outside this class, that is the bug this class exists to
 * prevent (a systematically wrong absolute time on every exported trace,
 * while internal Δt-based math still looks fine).
 *
 * For CsvReplaySource, `wallClockMillisAtAnchor` is simply the first row's
 * `timestamp_ms` (the CSV's own declared epoch time) and
 * `monotonicNanosAtAnchor` is that same row's synthesised timestampNanos --
 * so replay's ClockAnchor is an identity conversion by construction, which
 * is the correct behaviour: a replayed file's exported trace should carry
 * the SAME absolute times the file declared, not the wall-clock time replay
 * happened to run at.
 */
class ClockAnchor(
    private val monotonicNanosAtAnchor: Long,
    private val wallClockMillisAtAnchor: Long,
) {
    fun toUnixEpochMillis(timestampNanos: Long): Long =
        wallClockMillisAtAnchor + (timestampNanos - monotonicNanosAtAnchor) / 1_000_000L

    companion object {
        fun capturedNow(monotonicNanosNow: Long, wallClockMillisNow: Long) =
            ClockAnchor(monotonicNanosNow, wallClockMillisNow)
    }
}
