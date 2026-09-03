package org.anchor.prefilter

import org.anchor.math.Vec3

/**
 * CONTRACT GAP, flagged rather than silently worked around -- read before
 * assuming this class should not exist.
 *
 * contracts/frame_convention.md requires gravity as a distinct channel
 * (linear = raw_accel - gravity) and names two acceptable sources: the
 * device TYPE_GRAVITY sensor, or a low-pass filter on raw accel, and asks
 * implementers to confirm which one features.py actually used before
 * building this. features.py reads phone_gravity_x/y/z_mps2 directly from
 * IO-VNBD, which supplies real recorded gravity columns -- so training saw
 * the genuine sensor-fused estimate, not a low-pass approximation.
 *
 * The frozen contracts/replay_csv/schema.json 15 columns carry NO
 * gravity_x/y/z field at all (checked directly against the committed
 * schema, not assumed). That means ANY CsvReplaySource-backed session --
 * including every replay demo -- structurally cannot supply the same
 * gravity signal training saw. This class is the necessary fallback for
 * that gap, used for CSV replay unconditionally, and for live Android only
 * if TYPE_GRAVITY registration ever fails (it should not, on any real
 * device -- TYPE_GRAVITY is the preferred, more accurate source live and
 * is used directly by AndroidSensorSource when available, not routed
 * through this class).
 *
 * This is a genuine, acknowledged accuracy gap for replay specifically,
 * not a cosmetic one -- flagged in the Week-1 report as worth raising with
 * the ML track: either the replay_csv contract gains a gravity column (a
 * schema change, not mine to make unilaterally -- see the open ownership
 * question on contracts/replay_csv/ recorded this session), or replay-
 * sourced alignment is accepted as an approximation of what a live phone
 * or the training pipeline actually saw.
 *
 * alpha=0.8 matches the commonly-published default for this exact
 * accelerometer-only gravity extraction technique (an exponential low-pass
 * with a ~1-2s time constant at typical phone sampling rates) -- a
 * reasonable, standard starting point, not independently tuned here.
 */
class GravityLowPassEstimator(private val alpha: Double = 0.8) {
    private var estimate: Vec3? = null

    fun update(accelRaw: Vec3): Vec3 {
        val previous = estimate ?: accelRaw
        val next = Vec3(
            x = alpha * previous.x + (1.0 - alpha) * accelRaw.x,
            y = alpha * previous.y + (1.0 - alpha) * accelRaw.y,
            z = alpha * previous.z + (1.0 - alpha) * accelRaw.z,
        )
        estimate = next
        return next
    }

    fun reset() {
        estimate = null
    }
}
