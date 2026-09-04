package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Mat3
import org.anchor.math.Quaternion
import org.anchor.math.Vec
import org.anchor.math.Vec3

/**
 * FR-09: a 15-state error-state EKF over (position, velocity, attitude,
 * accelerometer bias, gyroscope bias). This class holds the 16-number
 * NominalState plus the 15x15 error covariance and exposes exactly two
 * operations: propagate() (IMU mechanization) and correct() (generic
 * measurement update, FR-10/11/26/27 and eventually GNSS all go through
 * this one method via MeasurementModel).
 *
 * Reference derivation: Sola, "Quaternion kinematics for the error-state
 * Kalman filter" (2017) -- the closest thing to a canonical reference for
 * exactly this algorithm, and consistent with the invariant-EKF approach
 * of Brossard et al.'s AI-IMU Dead-Reckoning, which the PRD itself cites
 * as the closest prior art to this filter.
 *
 * SCOPE NOTE, stated once here rather than repeated per method: this
 * implementation prioritises correctness and testability, per this
 * week's explicit instruction, over the zero-allocation hot-path
 * discipline PRD-ANDROID-ENGINE.md Section5.3 schedules for Week 3 (the
 * 30-minute p99.9 soak test is explicitly a Week-3 deliverable in the
 * PRD's own phasing, not a Week-2 gap). propagate()/correct() allocate
 * freely via Mat/Vec; that is a deliberate, sequenced choice, not an
 * oversight.
 */
class ErrorStateEkf(
    initialState: NominalState,
    initialCovariance: Mat,
    private val processNoise: ProcessNoiseConfig = ProcessNoiseConfig(),
) {
    init {
        require(initialCovariance.rows == ErrorStateLayout.DIM && initialCovariance.cols == ErrorStateLayout.DIM) {
            "initialCovariance must be ${ErrorStateLayout.DIM}x${ErrorStateLayout.DIM}, got ${initialCovariance.rows}x${initialCovariance.cols}"
        }
    }

    var state: NominalState = initialState
        private set

    var covariance: Mat = initialCovariance
        private set

    /**
     * IMU mechanization: integrates one accel+gyro sample over dt seconds.
     *
     * accelBody: RAW, gravity-INCLUSIVE specific force, vehicle/body frame,
     * m/s^2 -- NOT the gravity-removed linear acceleration ModelRunner
     * consumes. Standard strapdown mechanization removes gravity inside
     * this equation (rotate specific force into nav frame, then add the
     * nav-frame gravity vector back); feeding already-gravity-removed
     * input here would double-remove it. This is a deliberate difference
     * from org.anchor.alignment.VehicleFrameProjection's output, not an
     * oversight -- see this PR's design note.
     *
     * gyroBody: raw angular rate, vehicle/body frame, rad/s.
     */
    fun propagate(accelBody: Vec3, gyroBody: Vec3, dt: Double) {
        require(dt > 0.0) { "propagate() requires dt > 0, got $dt" }

        val correctedAccel = accelBody - state.accelBias
        val correctedGyro = gyroBody - state.gyroBias
        val rotation = state.orientation.toRotationMatrix()

        // ---- Nominal state propagation ----
        // Specific force rotated into nav frame, plus gravity added back --
        // this IS the gravity-compensation step; there is no other one.
        val accelNav = rotation * correctedAccel + GRAVITY_NAV

        // Second-order position update (exact under a constant-acceleration-
        // during-the-step assumption, which is reasonable at IMU rates of a
        // few ms) -- strictly more accurate than p += v*dt alone, at
        // negligible extra cost.
        val newPosition = state.position + state.velocity * dt + accelNav * (0.5 * dt * dt)
        val newVelocity = state.velocity + accelNav * dt

        // Exact exponential-map quaternion integration (not first-order
        // Euler) -- see Quaternion.fromRotationVector's own doc for why.
        val rotationIncrement = Quaternion.fromRotationVector(correctedGyro * dt)
        val newOrientation = (state.orientation * rotationIncrement).normalized()

        // Bias nominal values are NOT propagated deterministically -- they
        // evolve only through process noise (below) and are corrected only
        // by measurement updates. This is standard random-walk bias
        // modelling, not a missing term.
        state = NominalState(
            position = newPosition,
            velocity = newVelocity,
            orientation = newOrientation,
            accelBias = state.accelBias,
            gyroBias = state.gyroBias,
        )

        // ---- Error-state covariance propagation ----
        // Continuous-time error dynamics (Sola 2017 Section VI, "local"/
        // body-frame-composed error convention -- consistent with how
        // correct() injects a correction below, this MUST match):
        //   delta-p-dot     = delta-v
        //   delta-v-dot     = -R [a_corrected]_x delta-theta  -  R delta-b_a  -  R n_a
        //   delta-theta-dot = -[omega_corrected]_x delta-theta  -  delta-b_g  -  n_g
        //   delta-b_a-dot   = n_ba   (random walk)
        //   delta-b_g-dot   = n_bg   (random walk)
        // discretised as Phi ~= I + F_c*dt (first-order -- correct and
        // standard at IMU-rate dt; an exact Van Loan discretisation buys
        // nothing measurable here and is exactly the unneeded complexity
        // this week's instructions say to avoid).
        val fc = Mat.zeros(ErrorStateLayout.DIM, ErrorStateLayout.DIM)
        fc.setBlock3(ErrorStateLayout.POS, ErrorStateLayout.VEL, Mat3.IDENTITY)
        fc.setBlock3(ErrorStateLayout.VEL, ErrorStateLayout.THETA, (rotation * Mat.skew(correctedAccel)) * -1.0)
        fc.setBlock3(ErrorStateLayout.VEL, ErrorStateLayout.ACCEL_BIAS, rotation * -1.0)
        fc.setBlock3(ErrorStateLayout.THETA, ErrorStateLayout.THETA, Mat.skew(correctedGyro) * -1.0)
        fc.setBlock3(ErrorStateLayout.THETA, ErrorStateLayout.GYRO_BIAS, Mat3.IDENTITY * -1.0)

        val identity15 = Mat.identity(ErrorStateLayout.DIM)
        val phi = identity15 + (fc * dt)

        // Discrete process noise Q_d = G_c Q_c G_c^T * dt. G_c's four
        // nonzero 3x3 blocks are -R (into delta-v, driven by n_a), -I
        // (into delta-theta, driven by n_g), I (into delta-b_a, driven by
        // n_ba), I (into delta-b_g, driven by n_bg); Q_c is block-diagonal
        // (the four noise sources are independent). Because R is a
        // rotation matrix, R * (sigma^2 * I) * R^T == sigma^2 * I exactly
        // (rotating isotropic noise leaves it isotropic) -- so Q_d is
        // ITSELF exactly block-diagonal with these four scaled-identity
        // blocks, computed directly below rather than via a full G_c/Q_c
        // matrix multiply, which would reach the identical result through
        // more arithmetic on blocks that are zero by construction.
        val qd = Mat.zeros(ErrorStateLayout.DIM, ErrorStateLayout.DIM)
        val velNoiseVar = processNoise.accelNoiseDensity * processNoise.accelNoiseDensity * dt
        val thetaNoiseVar = processNoise.gyroNoiseDensity * processNoise.gyroNoiseDensity * dt
        val accelBiasVar = processNoise.accelBiasRandomWalk * processNoise.accelBiasRandomWalk * dt
        val gyroBiasVar = processNoise.gyroBiasRandomWalk * processNoise.gyroBiasRandomWalk * dt
        for (i in 0 until ErrorStateLayout.BLOCK) {
            qd[ErrorStateLayout.VEL + i, ErrorStateLayout.VEL + i] = velNoiseVar
            qd[ErrorStateLayout.THETA + i, ErrorStateLayout.THETA + i] = thetaNoiseVar
            qd[ErrorStateLayout.ACCEL_BIAS + i, ErrorStateLayout.ACCEL_BIAS + i] = accelBiasVar
            qd[ErrorStateLayout.GYRO_BIAS + i, ErrorStateLayout.GYRO_BIAS + i] = gyroBiasVar
        }

        covariance = ((phi * covariance) * phi.transpose() + qd).symmetrized()
    }

    /**
     * Generic measurement update (FR-09's Joseph-form requirement).
     * ErrorStateEkf has no knowledge of what [measurement] concretely is
     * -- NhcUpdate, ZuptUpdate, VelocityUpdate, and any future GNSS/
     * chi-square-gated update all arrive here through the exact same path.
     */
    fun correct(measurement: MeasurementModel) {
        val h = measurement.jacobian(state)
        require(h.rows == measurement.dimension && h.cols == ErrorStateLayout.DIM) {
            "${measurement.name} jacobian() must be ${measurement.dimension}x${ErrorStateLayout.DIM}, got ${h.rows}x${h.cols}"
        }
        val r = measurement.noise()
        require(r.rows == measurement.dimension && r.cols == measurement.dimension) {
            "${measurement.name} noise() must be ${measurement.dimension}x${measurement.dimension}, got ${r.rows}x${r.cols}"
        }

        val predicted = measurement.predicted(state)
        val actual = measurement.actual()
        val innovation = actual - predicted

        val ht = h.transpose()
        val innovationCovariance = (h * covariance) * ht + r
        val kalmanGain = (covariance * ht) * innovationCovariance.inverse()

        val correction = kalmanGain * innovation

        state = injectCorrection(state, correction)

        // Joseph form: P = (I-KH) P (I-KH)^T + K R K^T -- FR-09's own
        // requirement, algebraically equal to the simplified (I-KH)P for
        // an exact optimal K, but far more robust to the floating-point
        // rounding that WOULD otherwise erode positive-definiteness over
        // many update cycles.
        val identity15 = Mat.identity(ErrorStateLayout.DIM)
        val iMinusKh = identity15 - (kalmanGain * h)
        covariance = ((iMinusKh * covariance) * iMinusKh.transpose() + (kalmanGain * r) * kalmanGain.transpose()).symmetrized()
    }

    /**
     * Injects a 15-element error-state correction into the nominal state,
     * then implicitly resets the error state to zero (its expected value
     * is always zero by ESKF construction -- only its covariance persists,
     * which is why this method returns a NEW NominalState and nothing
     * carries the error vector forward).
     *
     * Deliberate, documented simplification: the "reset Jacobian" that
     * technically applies after injecting a large delta-theta (accounting
     * for the linearisation point shifting) is omitted -- standard
     * practice for small corrections (G ~= I), and what the AI-IMU
     * Dead-Reckoning paper this filter's design follows also does. Revisit
     * only if a synthetic test shows large single-step corrections in
     * practice.
     */
    private fun injectCorrection(s: NominalState, correction: Vec): NominalState {
        val dp = correction.slice3(ErrorStateLayout.POS)
        val dv = correction.slice3(ErrorStateLayout.VEL)
        val dtheta = correction.slice3(ErrorStateLayout.THETA)
        val dba = correction.slice3(ErrorStateLayout.ACCEL_BIAS)
        val dbg = correction.slice3(ErrorStateLayout.GYRO_BIAS)

        return NominalState(
            position = s.position + dp,
            velocity = s.velocity + dv,
            // Right-multiplicative (body-frame-composed) injection --
            // MUST match the "local" error convention F_c above was
            // derived under; left-multiplying here would silently mix two
            // different error conventions and corrupt the attitude estimate.
            orientation = (s.orientation * Quaternion.fromRotationVector(dtheta)).normalized(),
            accelBias = s.accelBias + dba,
            gyroBias = s.gyroBias + dbg,
        )
    }

    companion object {
        /** ENU nav frame: gravity points down, i.e. -z. */
        val GRAVITY_NAV = Vec3(0.0, 0.0, -9.80665)
    }
}
