package org.anchor.alignment

import org.anchor.math.Mat3
import org.anchor.math.Vec3

/**
 * The live/replay-streaming bridge around GravityAlign and YawResolver own
 * pure, batch-style functions. This is NOT a line-for-line port of
 * ml/anchor/data/features.py align_sequence_to_vehicle_frame -- it cannot
 * be, because that Python function operates on a whole finite recorded
 * sequence array at once, and a phone (or a replay session) sees samples
 * one at a time. The MATH inside is bit-comparable (GravityAlign,
 * YawResolver); the buffering strategy around it is new code, built to
 * match FR-04/FR-06 own framing: this is a per-mount-session calibration,
 * re-run on remount, not a per-sample computation -- contracts/
 * frame_convention.md own words, not a decision made here.
 *
 * Two-phase, not one: FR-04 accepts roll/pitch (levelling) from as little
 * as 3s of stationary or steady motion. Yaw resolution needs real driving
 * variance -- YawResolver own gate requires 200 valid paired samples with
 * reference std >= 1e-3, which a phone sitting still for 3-5s will not
 * produce. Collapsing these into one "5s calibrating" state would either
 * make yaw wait needlessly long past when levelling is ready, or falsely
 * claim yaw is resolved when YawResolver would have silently fallen back
 * to 0.0 (identity). Exposing them separately lets the engine apply
 * levelling immediately while correctly reporting yaw as unresolved.
 */
class AlignmentCalibrator(
    private val longitudinalReference: LongitudinalReference,
) {
    private val gravitySamples = ArrayList<Vec3>()
    private val accelRawSamples = ArrayList<Vec3>()

    /** gravity and accelRaw MUST be from the same sample tick, both in
     *  device/phone frame, both m/s^2. Order of addSample calls IS the
     *  order YawResolver own index-based LongitudinalReference will be
     *  queried against -- callers must add samples in timestamp order. */
    fun addSample(gravity: Vec3, accelRaw: Vec3) {
        gravitySamples.add(gravity)
        accelRawSamples.add(accelRaw)
    }

    fun sampleCount(): Int = gravitySamples.size

    /** Available as soon as ANY finite gravity sample exists (FR-04: as
     *  little as 3s stationary/steady). Corresponds to R_level in
     *  features.py, i.e. levelling only, yaw not yet applied. */
    fun levellingRotation(): Mat3 = GravityAlign.rotationGravityToDown(gravitySamples)

    /**
     * True once YawResolver own internal gate (>=200 valid paired samples,
     * reference std >= 1e-3) would actually attempt a real correlation
     * search rather than silently falling back to identity yaw. Callers
     * should treat a false here as "levelling-only alignment is live, yaw
     * is still 0.0 by construction, not yet a real estimate" -- exactly
     * the "AVOID FALSE MATCHES" caution the streaming split above exists
     * to enforce.
     */
    fun isYawResolvable(): Boolean {
        var valid = 0
        for (i in accelRawSamples.indices) {
            val ref = longitudinalReference.at(i) ?: continue
            if (ref.isFinite()) valid++
        }
        return valid >= 200
    }

    /** The combined level+yaw rotation, R in features.py notation. Safe to
     *  call at any buffer size -- yaw silently reports 0.0 (identity) below
     *  YawResolver own gate, matching the Python fallback exactly, not an
     *  extra behaviour invented here. */
    fun currentRotation(): Mat3 {
        val rLevel = levellingRotation()
        val linear = accelRawSamples.indices.map { i -> accelRawSamples[i] - gravitySamples[i] }
        val levelled = linear.map { rLevel * it }
        val levelledXY = levelled.map { it.x to it.y }
        val theta = YawResolver.bestYaw(levelledXY, longitudinalReference)
        return YawResolver.yawRotation(theta) * rLevel
    }

    fun reset() {
        gravitySamples.clear()
        accelRawSamples.clear()
    }
}

/**
 * Direct port of the per-sample half of features.py sequence_model_features:
 *   lin_v = (accel - grav) @ R.T
 *   gyro_v = gyro @ R.T
 * applied here to one sample against an already-fit rotation, which is the
 * live/replay equivalent of that batch operation applied per-row.
 */
object VehicleFrameProjection {
    fun project(rotation: Mat3, accelRaw: Vec3, gravity: Vec3, gyroRaw: Vec3): Pair<Vec3, Vec3> {
        val linearVehicle = rotation * (accelRaw - gravity)
        val gyroVehicle = rotation * gyroRaw
        return linearVehicle to gyroVehicle
    }
}
