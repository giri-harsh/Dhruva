package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/** NOTE: written, not run via Gradle -- see the Week-1 report for why. */
class ZuptUpdateTest {

    @Test
    fun `jacobian directly selects the velocity error block, identity, nothing else`() {
        val h = ZuptUpdate().jacobian(NominalState.zero())
        assertEquals(3, h.rows)
        for (i in 0 until 3) {
            for (c in 0 until ErrorStateLayout.DIM) {
                val expected = if (c == ErrorStateLayout.VEL + i) 1.0 else 0.0
                assertEquals(expected, h[i, c], 1e-12, "H[$i,$c]")
            }
        }
    }

    @Test
    fun `does not depend on orientation, unlike NHC`() {
        // ZUPT observes nav-frame velocity directly -- its Jacobian must
        // be identical regardless of the current attitude, unlike NHC's
        // (which rotates through R^T). A meaningful distinguishing test
        // between the two update types, not a restatement of the same fact.
        val level = ZuptUpdate().jacobian(NominalState.zero())
        val tilted = ZuptUpdate().jacobian(
            NominalState.zero().copy(
                orientation = org.anchor.math.Quaternion.fromRotationVector(Vec3(0.3, 0.2, 0.1)),
            ),
        )
        for (i in 0 until 3) for (c in 0 until ErrorStateLayout.DIM) assertEquals(level[i, c], tilted[i, c], 1e-12)
    }

    @Test
    fun `120 seconds of ZUPT-corrected simulated idle keeps the marker from creeping`() {
        // FR-26's own acceptance criterion, in miniature: velocity is
        // corrected to zero repeatedly (as an orchestrator would during a
        // detected-stationary period), so position must not drift despite
        // continuous propagation over the same window.
        val ekf = ErrorStateEkf(NominalState.zero(), diagonalCovariance())
        val g = ErrorStateEkf.GRAVITY_NAV.norm()
        val dt = 0.1
        val steps = 1200 // 120 s at 10 Hz

        repeat(steps) {
            // A stationary phone's real accelerometer is not perfectly
            // noise-free -- inject a tiny, deterministic wobble so this is
            // a genuine test of ZUPT counteracting drift, not a repeat of
            // the exact-cancellation stationary case already covered in
            // ErrorStateEkfPropagationTest.
            val wobble = 0.002 * kotlin.math.sin(it * 0.37)
            ekf.propagate(Vec3(wobble, 0.0, g), Vec3(0.0005, 0.0, 0.0), dt)
            if (it % 5 == 0) ekf.correct(ZuptUpdate())
        }

        assertTrue(kotlin.math.abs(ekf.state.position.x) < 1.0, "position drifted ${ekf.state.position.x} m over 120s of ZUPT-corrected idle")
    }

    private fun diagonalCovariance(): Mat {
        val diag = DoubleArray(ErrorStateLayout.DIM) { 0.01 }
        return Mat.diagonal(diag)
    }
}
