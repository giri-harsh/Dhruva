package org.anchor.fusion

import org.anchor.math.Quaternion
import org.anchor.math.Vec3

/**
 * The 16-number nominal state FR-09 names as "position, velocity,
 * attitude, accel bias, gyro bias" -- 16, not 15, because the quaternion
 * carries a redundant 4th parameter (unit-norm constraint); the 15-wide
 * ERROR state (ErrorStateLayout) is what FR-09 actually counts.
 *
 * Frames (org.anchor.fusion package convention, documented once here):
 *   position, velocity  -- navigation frame, local ENU tangent plane.
 *     Not fixed by contracts/frame_convention.md (that file fixes the
 *     VEHICLE body frame only) -- chosen here because vehicle-frame z=up
 *     already matches ENU z=up, avoiding a body/nav axis-flip convention
 *     to track through every rotation in this package.
 *   orientation -- body-to-nav quaternion (Quaternion.rotate(v_body) ==
 *     v_nav), continuing the exact directional convention
 *     org.anchor.alignment already uses for Mat3.
 *   accelBias, gyroBias -- body (vehicle) frame, contracts/frame_convention.md
 *     axes (x=forward, y=left, z=up).
 */
data class NominalState(
    val position: Vec3,
    val velocity: Vec3,
    val orientation: Quaternion,
    val accelBias: Vec3,
    val gyroBias: Vec3,
) {
    /** velocity expressed in body frame -- what NHC, ZUPT and
     *  VelocityUpdate all actually observe. R(q)^T == R(q).conjugate()
     *  for a unit quaternion (nav-to-body). */
    fun bodyVelocity(): Vec3 = orientation.conjugate().rotate(velocity)

    companion object {
        fun zero() = NominalState(
            position = Vec3.ZERO,
            velocity = Vec3.ZERO,
            orientation = Quaternion.IDENTITY,
            accelBias = Vec3.ZERO,
            gyroBias = Vec3.ZERO,
        )
    }
}
