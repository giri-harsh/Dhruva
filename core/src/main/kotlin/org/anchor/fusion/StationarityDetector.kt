package org.anchor.fusion

import org.anchor.math.Vec3

/**
 * FR-26's stationarity trigger has two halves: "IMU energy below the
 * stationarity threshold AND near-zero predicted displacement." This
 * class implements only the IMU-energy half -- the model-displacement
 * half needs a live ModelRunner stream feeding a sliding window, which
 * does not exist yet (no EngineOrchestrator this week; ModelRunner is
 * proven correct in isolation via Week 1's golden-vector test, not wired
 * into a live pipeline). Wiring that second half is a clean, later
 * extension: whoever builds the orchestrator ANDs this class own
 * isStationary() with a "|model displacement| < threshold" check.
 *
 * Energy metric: trace of the sample covariance of the accel VECTOR (sum
 * of per-axis variance from the window's own mean), and likewise for
 * gyro. Deliberately NOT variance-of-the-scalar-magnitude -- an earlier
 * version used |accel| - g, which is all but blind to horizontal
 * vibration once gravity dominates the z-axis (sqrt(g^2 + small_x^2) is
 * nearly flat in small_x near g, since the norm is a saturating function
 * of a small perturbation on a large baseline) -- a real bug caught by
 * StationarityDetectorTest actually failing, not found by inspection.
 * Per-axis variance has no such blind spot: wobble on any axis
 * contributes to the sum regardless of gravity's own magnitude.
 */
class StationarityDetector(
    private val windowSize: Int = 20,
    private val accelVarianceThreshold: Double = DEFAULT_ACCEL_VARIANCE_THRESHOLD,
    private val gyroVarianceThreshold: Double = DEFAULT_GYRO_VARIANCE_THRESHOLD,
) {
    private val accelSamples = ArrayDeque<Vec3>()
    private val gyroSamples = ArrayDeque<Vec3>()

    init {
        require(windowSize >= 2) { "windowSize must be >= 2 to compute a variance, got $windowSize" }
    }

    /** accelBody: raw, gravity-inclusive specific force (same convention
     *  as ErrorStateEkf.propagate()'s input), m/s^2. gyroBody: rad/s. */
    fun addSample(accelBody: Vec3, gyroBody: Vec3): Boolean {
        accelSamples.addLast(accelBody)
        gyroSamples.addLast(gyroBody)
        if (accelSamples.size > windowSize) accelSamples.removeFirst()
        if (gyroSamples.size > windowSize) gyroSamples.removeFirst()
        return isStationary()
    }

    fun isStationary(): Boolean {
        if (accelSamples.size < windowSize) return false // not enough history yet
        return vectorEnergy(accelSamples) < accelVarianceThreshold && vectorEnergy(gyroSamples) < gyroVarianceThreshold
    }

    fun reset() {
        accelSamples.clear()
        gyroSamples.clear()
    }

    /** Sum of per-axis variance from the window's own mean vector (the
     *  trace of the 3x3 sample covariance) -- see class doc for why this,
     *  not a scalar-magnitude variance. */
    private fun vectorEnergy(samples: ArrayDeque<Vec3>): Double {
        val meanX = samples.sumOf { it.x } / samples.size
        val meanY = samples.sumOf { it.y } / samples.size
        val meanZ = samples.sumOf { it.z } / samples.size
        return samples.sumOf {
            val dx = it.x - meanX
            val dy = it.y - meanY
            val dz = it.z - meanZ
            dx * dx + dy * dy + dz * dz
        } / samples.size
    }

    companion object {
        const val GRAVITY_MAGNITUDE = 9.80665

        /** (m/s^2)^2, summed across 3 axes -- [VERIFY] against real
         *  device vibration spectra, same discipline as FR-02's own
         *  unmeasured attenuation target. */
        const val DEFAULT_ACCEL_VARIANCE_THRESHOLD = 0.02

        /** (rad/s)^2, summed across 3 axes -- [VERIFY]. */
        const val DEFAULT_GYRO_VARIANCE_THRESHOLD = 0.001
    }
}
