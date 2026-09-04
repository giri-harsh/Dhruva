package org.anchor.fusion

import org.anchor.math.Mat

/**
 * Shared derivation used by both NhcUpdate and VelocityUpdate: both
 * observe a component of body-frame velocity v_body = R(q)^T v_nav, they
 * just select different rows of the identical 3x15 Jacobian (NHC: rows
 * y,z; VelocityUpdate: row x, the forward axis). Deriving this once and
 * having both call it, rather than each re-deriving it, is what makes
 * that shared structure visible in the code instead of being an
 * unremarked coincidence between two files.
 *
 * Derivation (first-order perturbation of v_body around the nominal
 * state, "local"/body-frame error convention, matching ErrorStateEkf):
 *   v_body = R(q)^T v_nav
 *          = [(R_nom)(I + [dtheta]_x)]^T (v_nav_nom + dv)
 *          = (I - [dtheta]_x) R_nom^T (v_nav_nom + dv)          {[dtheta]_x^T = -[dtheta]_x}
 *   ~= v_body_nom + R_nom^T dv - [dtheta]_x R_nom^T v_nav_nom   {drop second-order dtheta*dv}
 *    = v_body_nom + R_nom^T dv + [v_body_nom]_x dtheta          {-[a]_x b == [b]_x a}
 * so:
 *   d(v_body)/d(dp)     = 0
 *   d(v_body)/d(dv)     = R_nom^T
 *   d(v_body)/d(dtheta) = [v_body_nom]_x
 *   d(v_body)/d(db_a)   = 0
 *   d(v_body)/d(db_g)   = 0
 */
object BodyVelocityJacobian {
    fun full3x15(state: NominalState): Mat {
        val h = Mat.zeros(3, ErrorStateLayout.DIM)
        val rTransposed = state.orientation.toRotationMatrix().transpose()
        val bodyVel = state.bodyVelocity()
        h.setBlock3(0, ErrorStateLayout.VEL, rTransposed)
        h.setBlock3(0, ErrorStateLayout.THETA, Mat.skew(bodyVel))
        return h
    }

    /** Rows selected from full3x15, e.g. rows=[1,2] for NHC (y,z), [0] for
     *  VelocityUpdate's forward axis. */
    fun selectRows(state: NominalState, rows: IntArray): Mat {
        val full = full3x15(state)
        val h = Mat.zeros(rows.size, ErrorStateLayout.DIM)
        for ((outRow, srcRow) in rows.withIndex()) {
            for (c in 0 until ErrorStateLayout.DIM) h[outRow, c] = full[srcRow, c]
        }
        return h
    }
}
