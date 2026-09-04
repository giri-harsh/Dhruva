package org.anchor.alignment

import org.anchor.math.Mat3
import org.anchor.math.Vec3
import kotlin.math.sqrt

/**
 * Direct port of ml/anchor/data/features.py _rotation_gravity_to_down.
 * Computes the rotation R such that R times (mean gravity vector) is
 * parallel to (0, 0, -1) -- i.e. levels the device so vehicle-frame down
 * really is down, fixing 2 of 3 degrees of freedom (roll and pitch) from
 * gravity alone. This is FR-04 own algorithm, not a reinterpretation of it.
 *
 * Every branch below (empty input, near-zero gravity magnitude, the
 * antiparallel special case) is reproduced exactly, including the
 * seemingly-arbitrary diag(1,-1,-1) fallback for the antiparallel case --
 * that specific choice is what the Python reference does, and matching it
 * exactly matters more here than deriving a different, equally-valid
 * alternative, per the instruction not to invent an alternative
 * preprocessing pipeline.
 *
 * The mean/rotation-fit step here operates on a bounded, already-collected
 * sample buffer -- this is a per-mount-session calibration (frozen in
 * contracts/frame_convention.md as "not a per-sample computation"), not a
 * per-tick recomputation. See AlignmentCalibrator for how a live/replay
 * session actually accumulates that buffer -- features.py operates on a
 * whole finite recorded sequence at once, which this pure function
 * mirrors; the live-streaming buffering strategy around it is necessarily
 * new code, not a literal port, because no live-streaming equivalent
 * exists in features.py to port from.
 */
object GravityAlign {

    fun rotationGravityToDown(gravitySamples: List<Vec3>): Mat3 {
        val finite = gravitySamples.filter { it.x.isFinite() && it.y.isFinite() && it.z.isFinite() }
        if (finite.isEmpty()) return Mat3.IDENTITY

        val meanX = finite.sumOf { it.x } / finite.size
        val meanY = finite.sumOf { it.y } / finite.size
        val meanZ = finite.sumOf { it.z } / finite.size
        val gm = Vec3(meanX, meanY, meanZ)
        val n = gm.norm()
        if (n < 1e-6) return Mat3.IDENTITY

        val src = Vec3(gm.x / n, gm.y / n, gm.z / n)
        val dst = Vec3(0.0, 0.0, -1.0)

        val v = cross(src, dst)
        val c = dot(src, dst)

        if (v.norm() < 1e-9) {
            return if (c > 0) Mat3.IDENTITY else Mat3.diagonal(1.0, -1.0, -1.0)
        }

        val vx = Mat3(
            0.0, -v.z, v.y,
            v.z, 0.0, -v.x,
            -v.y, v.x, 0.0,
        )
        val vx2 = vx * vx
        return Mat3.IDENTITY + vx + (vx2 * (1.0 / (1.0 + c)))
    }

    private fun cross(a: Vec3, b: Vec3) = Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )

    private fun dot(a: Vec3, b: Vec3) = a.x * b.x + a.y * b.y + a.z * b.z
}
