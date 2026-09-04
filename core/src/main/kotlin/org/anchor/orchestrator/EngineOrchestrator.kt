package org.anchor.orchestrator

import org.anchor.alignment.GravityAlign
import org.anchor.fusion.ChiSquareGate
import org.anchor.fusion.ErrorStateEkf
import org.anchor.fusion.GatedMeasurementUpdate
import org.anchor.fusion.GnssPositionUpdate
import org.anchor.fusion.NhcUpdate
import org.anchor.fusion.NominalState
import org.anchor.fusion.StationarityDetector
import org.anchor.fusion.ZuptUpdate
import org.anchor.math.Mat
import org.anchor.math.Mat3
import org.anchor.math.Vec3
import org.anchor.prefilter.GravityLowPassEstimator

enum class EngineMode { CALIBRATING, STATIONARY, DRIVING }

sealed interface EngineTickResult {
    /** Still buffering samples for alignment; the EKF has not been
     *  touched yet -- there is no valid phone-to-body rotation to convert
     *  a raw phone-frame reading into what propagate() requires. */
    data class Calibrating(val elapsedNanos: Long, val remainingNanos: Long) : EngineTickResult

    /** One propagate() + one mode-dependent correct() happened this tick.
     *  appliedUpdate is the applied MeasurementModel's own `name` (e.g.
     *  "ZUPT"/"NHC") -- MODE_EVENT-style, matching how that field is
     *  documented to be used elsewhere in `fusion`. */
    data class Propagated(val mode: EngineMode, val appliedUpdate: String) : EngineTickResult
}

/**
 * Coordinates the pipeline stages the Week-3 brief names: alignment/frame
 * conversion -> ESKF propagation -> mode-dependent measurement update.
 * Owns SEQUENCING ONLY. Every number this class touches is computed by an
 * existing, independently-tested component (GravityAlign, ErrorStateEkf,
 * StationarityDetector, NhcUpdate/ZuptUpdate, GatedMeasurementUpdate) --
 * this class decides WHEN to call each and WHICH one, not HOW any of them
 * work. ErrorStateEkf itself is not modified by this class or this PR.
 *
 * Two phases, matching how alignment is documented elsewhere in this
 * codebase as a per-mount-session calibration, not a per-sample
 * computation (see AlignmentCalibrator's own doc):
 *  - CALIBRATING: buffers a running gravity estimate; the EKF is not
 *    touched at all yet, because there is no valid rotation yet to turn a
 *    phone-frame reading into the body-frame ErrorStateEkf.propagate()
 *    requires.
 *  - RUNNING (STATIONARY/DRIVING): the rotation is frozen for the rest of
 *    the session (re-calibrating mid-session is FR-06/RemountDetector's
 *    job, not built this week -- see this PR's known limitations); every
 *    sample is rotated into body frame and propagated.
 *
 * ## A genuine sign-convention gap found by wiring these two subsystems
 * together for the first time (documented here, not silently patched):
 *
 * `GravityAlign`/`AlignmentCalibrator` (Week 1) exist solely to feed the
 * ML model's GRAVITY-REMOVED linear-acceleration input (contracts/
 * frame_convention.md's Stage 1 -> Stage 2 step) -- a computation that is
 * structurally blind to which way "up" is, because subtracting the mean
 * gravity erases that information from the signal entirely. Feeding
 * `GravityAlign.rotationGravityToDown` the RAW gravity/accel reading (the
 * reaction-force convention both `contracts/frame_convention.md` line 16
 * and `ml/anchor/data/features.py` confirm: `accel_z`/`gravity_z` ~= +9.8
 * at rest) produces a rotation R such that `R @ (0,0,+9.8)` lands near
 * (0,0,-9.8) -- correct and verified for its own purpose (GravityAlignTest
 * checks exactly this), but the WRONG sign for `ErrorStateEkf.propagate()`,
 * which needs body-frame GRAVITY-INCLUSIVE specific force with POSITIVE
 * body-z at rest (algebra: `accelNav = R(q)*correctedAccel + GRAVITY_NAV`
 * must be ~0 when stationary and level, `GRAVITY_NAV=(0,0,-9.80665)`, so
 * `correctedAccel` must be `(0,0,+9.80665)` -- exactly what every Week-2
 * ESKF test already assumes, e.g. ZuptUpdateTest's own `Vec3(_, _, g)`).
 *
 * Nobody was wrong: each subsystem is correct for the one consumer it was
 * built for. Composing them for a THIRD consumer (gravity-inclusive
 * strapdown mechanization) that neither was designed against is new
 * territory `contracts/frame_convention.md` does not cover. The fix here
 * is local and minimal: feed `GravityAlign.rotationGravityToDown` the
 * NEGATED gravity estimate (the true, physical, points-down convention
 * the function's own name and docstring literally describe) to get a
 * rotation with the correct sign for THIS consumer, and verify that
 * choice empirically against ErrorStateEkf's own convention rather than
 * assume it (see the `check()` below) -- no Week 1 or Week 2 file is
 * touched. Yaw resolution (FR-05) is deliberately NOT composed into this
 * rotation this week: with no real LongitudinalReference wired (none is
 * built this week -- see known limitations), YawResolver's own documented
 * fallback makes yaw exactly 0.0/identity, so heading comes entirely from
 * gyro integration inside propagate() for now.
 */
class EngineOrchestrator(
    initialCovariance: Mat,
    private val config: EngineConfig = EngineConfig(),
    initialState: NominalState = NominalState.zero(),
) {
    private val ekf = ErrorStateEkf(initialState, initialCovariance, config.processNoise)
    private val gravityEstimator = GravityLowPassEstimator()
    private val stationarityDetector = StationarityDetector(windowSize = config.stationarityWindowSize)
    private val zuptUpdate = ZuptUpdate()
    private val nhcUpdate = NhcUpdate()

    private var calibrationStartNanos: Long? = null
    private var lastGravityEstimate: Vec3 = Vec3.ZERO
    private var frozenRotation: Mat3? = null
    private var lastTimestampNanos: Long? = null

    val state: NominalState get() = ekf.state
    val covariance: Mat get() = ekf.covariance

    val mode: EngineMode
        get() = when {
            frozenRotation == null -> EngineMode.CALIBRATING
            stationarityDetector.isStationary() -> EngineMode.STATIONARY
            else -> EngineMode.DRIVING
        }

    /**
     * One IMU sample: raw, phone/device frame, gravity-inclusive accel
     * (Stage 1, contracts/frame_convention.md) and raw gyro, both exactly
     * as SensorSource emits them. dt is derived from consecutive
     * timestamps, never a wall-clock read -- this keeps replay and live
     * operation on the identical code path (FR-18).
     */
    fun onImuSample(accelRawPhoneFrame: Vec3, gyroRawPhoneFrame: Vec3, timestampNanos: Long): EngineTickResult {
        val previousTimestamp = lastTimestampNanos
        require(previousTimestamp == null || timestampNanos >= previousTimestamp) {
            "timestampNanos went backwards ($timestampNanos < $previousTimestamp) -- " +
                "SensorSource's own contract requires non-decreasing event order"
        }
        val dt = previousTimestamp?.let { (timestampNanos - it) / 1_000_000_000.0 } ?: 0.0
        lastTimestampNanos = timestampNanos

        val rotation = frozenRotation ?: run {
            lastGravityEstimate = gravityEstimator.update(accelRawPhoneFrame)
            // Keep StationarityDetector's window warm across the
            // CALIBRATING -> RUNNING boundary, not just during RUNNING: its
            // energy metric is the trace of a sample covariance, which is
            // invariant under any orthogonal (rotation) transform --
            // trace(R@Sigma@R^T) == trace(Sigma) -- so feeding it the raw,
            // not-yet-rotated phone-frame reading here is exactly as valid
            // as feeding it accelBody/gyroBody after RUNNING starts. Without
            // this, the first windowSize-1 RUNNING ticks would fall through
            // isStationary()'s "not enough history yet" default of false and
            // misclassify as DRIVING immediately after a vehicle FR-04 just
            // spent 3s confirming was stationary or steady.
            stationarityDetector.addSample(accelRawPhoneFrame, gyroRawPhoneFrame)
            val start = calibrationStartNanos ?: timestampNanos.also { calibrationStartNanos = it }
            val elapsed = timestampNanos - start
            if (elapsed >= config.calibrationDurationNanos) {
                frozenRotation = freezeEskfRotation(lastGravityEstimate)
            }
            return EngineTickResult.Calibrating(
                elapsedNanos = elapsed,
                remainingNanos = (config.calibrationDurationNanos - elapsed).coerceAtLeast(0L),
            )
        }

        val accelBody = rotation * accelRawPhoneFrame
        val gyroBody = rotation * gyroRawPhoneFrame
        ekf.propagate(accelBody, gyroBody, dt)

        val isStationary = stationarityDetector.addSample(accelBody, gyroBody)
        return if (isStationary) {
            ekf.correct(zuptUpdate)
            EngineTickResult.Propagated(EngineMode.STATIONARY, zuptUpdate.name)
        } else {
            ekf.correct(nhcUpdate)
            EngineTickResult.Propagated(EngineMode.DRIVING, nhcUpdate.name)
        }
    }

    /**
     * GNSS entry point, deliberately scoped exactly like GnssPositionUpdate
     * itself: `positionEnu` must already be in the filter's local ENU nav
     * frame. Converting a raw lat/lon fix into that frame needs a chosen
     * tangent-plane origin and a real geodetic projection -- genuinely not
     * built this week (see GnssPositionUpdate's own doc and this PR's
     * known limitations); this method does not silently invent one.
     * `noiseVariance` is this week's stand-in for FR-12's fix-quality
     * classification: the caller expresses trust as measurement variance
     * (the same pattern VelocityUpdate/FR-11 already uses), since a real
     * CN0/satellite-count/accuracy-based classifier needs fields neither
     * SensorEvent.GnssFix nor contracts/replay_csv/schema.json carry
     * today.
     */
    fun applyGnssObservation(positionEnu: Vec3, noiseVariance: Vec3): ChiSquareGate.Result {
        check(frozenRotation != null) { "cannot apply a GNSS observation before alignment calibration completes" }
        val update = GnssPositionUpdate(positionEnu, noiseVariance)
        return GatedMeasurementUpdate.apply(ekf, update, config.gnssGateConfidence)
    }

    private fun freezeEskfRotation(gravityEstimate: Vec3): Mat3 {
        val rotation = GravityAlign.rotationGravityToDown(listOf(gravityEstimate * -1.0))
        val restSpecificForceZ = (rotation * gravityEstimate).z
        check(restSpecificForceZ > 0.0) {
            "alignment produced non-positive body-z specific force at rest ($restSpecificForceZ) -- " +
                "ErrorStateEkf's GRAVITY_NAV convention requires positive body-z here " +
                "(see this class's own doc comment on the gravity sign reconciliation); " +
                "the negation in freezeEskfRotation() needs re-checking, not the caller"
        }
        return rotation
    }
}
