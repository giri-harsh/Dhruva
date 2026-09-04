package org.anchor.fusion

/**
 * The 15-element error-state layout (FR-09), named once here so no other
 * file hardcodes an offset as a bare integer. Every block is 3-wide:
 *   [0:3)  δp   position error, nav frame (ENU), metres
 *   [3:6)  δv   velocity error, nav frame (ENU), m/s
 *   [6:9)  δθ   attitude error, body-frame minimal rotation vector, radians
 *   [9:12) δb_a accelerometer bias error, body frame, m/s^2
 *   [12:15)δb_g gyroscope bias error, body frame, rad/s
 *
 * Nominal-state note: the quaternion contributes only these 3 δθ error-DOF
 * despite being 4 numbers itself (1 unit-norm constraint removes one DOF)
 * -- this gap between a 16-wide nominal state and a 15-wide error state is
 * the defining feature of an error-state (as opposed to a naive quaternion)
 * EKF, not an inconsistency.
 */
object ErrorStateLayout {
    const val DIM = 15

    const val POS = 0
    const val VEL = 3
    const val THETA = 6
    const val ACCEL_BIAS = 9
    const val GYRO_BIAS = 12

    const val BLOCK = 3
}
