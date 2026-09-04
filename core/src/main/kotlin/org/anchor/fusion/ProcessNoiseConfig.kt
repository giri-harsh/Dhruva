package org.anchor.fusion

/**
 * Continuous-time process noise power spectral densities. These are
 * genuinely device-specific and unmeasured on real hardware yet --
 * [VERIFY], same discipline the PRD applies to its own unmeasured
 * numbers (v3 PRD Section1.2: "measure real bias instability on our
 * target devices -- published MEMS figures vary by 1-2 orders of
 * magnitude"). The defaults below are placeholder order-of-magnitude
 * values for a consumer-grade MEMS IMU, explicitly not to be trusted for
 * a real accuracy claim -- they exist so propagate() has *something*
 * principled to run synthetic tests against, not a stand-in for
 * calibration.
 *
 * Units, matching contracts/units.md throughout:
 *   accelNoiseDensity      (m/s^2) / sqrt(Hz)  -- velocity random walk driver
 *   gyroNoiseDensity       (rad/s) / sqrt(Hz)  -- attitude random walk driver
 *   accelBiasRandomWalk    (m/s^2) / sqrt(Hz)  -- accel bias instability driver
 *   gyroBiasRandomWalk     (rad/s) / sqrt(Hz)  -- gyro bias instability driver
 */
data class ProcessNoiseConfig(
    val accelNoiseDensity: Double = 1.0e-3,
    val gyroNoiseDensity: Double = 1.0e-4,
    val accelBiasRandomWalk: Double = 1.0e-5,
    val gyroBiasRandomWalk: Double = 1.0e-6,
) {
    init {
        require(accelNoiseDensity > 0.0) { "accelNoiseDensity must be positive" }
        require(gyroNoiseDensity > 0.0) { "gyroNoiseDensity must be positive" }
        require(accelBiasRandomWalk > 0.0) { "accelBiasRandomWalk must be positive" }
        require(gyroBiasRandomWalk > 0.0) { "gyroBiasRandomWalk must be positive" }
    }
}
