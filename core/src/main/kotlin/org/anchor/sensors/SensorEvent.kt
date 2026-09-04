package org.anchor.sensors

import org.anchor.math.Vec3

/**
 * The canonical event union both AndroidSensorSource and CsvReplaySource
 * emit. This IS the "same downstream pipeline" abstraction FR-18 depends
 * on: nothing past this type may know or care whether a sample came from a
 * live phone or a replayed CSV.
 *
 * timestampNanos is a MONOTONIC clock reading, meaningful only for
 * computing deltas within one continuous session -- never a wall-clock
 * value. This is FR-33's rule made structural rather than a comment:
 *   - AndroidSensorSource: `android.hardware.SensorEvent.timestamp` verbatim
 *     (nanoseconds since an arbitrary, device-specific monotonic epoch --
 *     NOT Unix time, per the Android platform docs).
 *   - CsvReplaySource: `timestamp_ms * 1_000_000L` from the CSV's own
 *     `timestamp_ms` column. The CSV's ms-since-Unix-epoch value is treated
 *     purely as a self-consistent monotonic timeline for a replay session,
 *     never compared against real wall-clock time.
 * Converting either of these to a genuine Unix-epoch millisecond value (for
 * TripExporter or any exported trace) is ClockAnchor's job, applied at
 * exactly one boundary -- never inline where an event is produced or
 * consumed.
 *
 * Every raw accelerometer/gyroscope/magnetometer sample here is Stage 1 per
 * contracts/frame_convention.md: phone-frame, gravity-inclusive for accel.
 * Nothing in this file performs alignment or gravity removal -- that is
 * org.anchor.alignment's job, strictly downstream of this type.
 */
sealed interface SensorEvent {
    val timestampNanos: Long

    /** m/s^2, phone/device frame, gravity-INCLUSIVE (Stage 1, raw). */
    data class Accelerometer(
        override val timestampNanos: Long,
        val value: Vec3,
    ) : SensorEvent

    /**
     * m/s^2, phone/device frame -- the device's own gravity estimate.
     * Live Android: TYPE_GRAVITY, delivered as its own sensor stream.
     * CSV replay: schema.json's 15 columns carry NO gravity channel, so
     * this event is never produced directly by CsvReplaySource -- it is
     * synthesised from the Accelerometer stream by
     * org.anchor.prefilter.GravityLowPassEstimator instead. See that
     * class's doc comment for why this is a real, flagged gap rather than
     * a silent substitution.
     */
    data class Gravity(
        override val timestampNanos: Long,
        val value: Vec3,
    ) : SensorEvent

    /** rad/s, phone/device frame, raw (Stage 1). Asserted |x|,|y|,|z| < 10
     *  rad/s at the point of construction -- see init block below, matching
     *  contracts/units.md's mandatory gyro sanity check. */
    data class Gyroscope(
        override val timestampNanos: Long,
        val value: Vec3,
    ) : SensorEvent {
        init {
            require(kotlin.math.abs(value.x) < 10.0 && kotlin.math.abs(value.y) < 10.0 && kotlin.math.abs(value.z) < 10.0) {
                "gyro |value| >= 10 rad/s (~573 deg/s) at t=$timestampNanos ns -- " +
                    "this is almost certainly a deg/s-vs-rad/s unit bug " +
                    "(contracts/units.md), not a real vehicle motion. value=$value"
            }
        }
    }

    /** microtesla, phone/device frame, raw. */
    data class Magnetometer(
        override val timestampNanos: Long,
        val value: Vec3,
    ) : SensorEvent

    /**
     * Present ONLY when a real fix exists. There is no "invalid fix" variant
     * carrying zeroed fields -- schema.json is explicit that gnss_valid=0
     * means the four gnss_* columns are empty, not zero, and 0,0 is a real
     * coordinate (off West Africa). The absence of a GnssFix event for a
     * given tick IS the "no fix" signal.
     */
    data class GnssFix(
        override val timestampNanos: Long,
        val latDeg: Double,
        val lonDeg: Double,
        val speedMps: Double,
        val courseDeg: Double,
    ) : SensorEvent {
        init {
            require(latDeg in -90.0..90.0) { "gnss_lat out of range: $latDeg" }
            require(lonDeg in -180.0..180.0) { "gnss_lon out of range: $lonDeg" }
            require(courseDeg in 0.0..360.0) { "gnss_course_deg out of range: $courseDeg" }
        }
    }
}
