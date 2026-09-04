package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec
import org.anchor.math.Vec3
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Phase 3.A-D: synthetic-trajectory propagation validation. Every
 * assertion here is a physically meaningful check against a hand-derived
 * expected value, not "propagate() ran without throwing."
 *
 * NOTE: written, not run via Gradle -- see the Week-1 report for why;
 * verified via direct kotlinc/junit-platform-console-standalone instead
 * (see this PR's own validation section).
 */
class ErrorStateEkfPropagationTest {

    private val g = ErrorStateEkf.GRAVITY_NAV.norm()

    private fun modestCovariance(): Mat {
        // A small, physically reasonable initial P -- diagonal, position/
        // velocity/attitude/bias blocks each at a plausible starting
        // uncertainty. Not [VERIFY]-tagged like ProcessNoiseConfig's
        // defaults because this value is a TEST FIXTURE choice, not a
        // claim about a real device.
        val diag = DoubleArray(ErrorStateLayout.DIM)
        for (i in 0 until 3) diag[ErrorStateLayout.POS + i] = 1.0        // 1 m^2
        for (i in 0 until 3) diag[ErrorStateLayout.VEL + i] = 0.25       // 0.5 m/s std
        for (i in 0 until 3) diag[ErrorStateLayout.THETA + i] = 0.01     // ~5.7 deg std
        for (i in 0 until 3) diag[ErrorStateLayout.ACCEL_BIAS + i] = 0.001
        for (i in 0 until 3) diag[ErrorStateLayout.GYRO_BIAS + i] = 0.0001
        return Mat.diagonal(diag)
    }

    private fun assertSymmetricPositiveDiagonal(p: Mat, context: String) {
        for (i in 0 until p.rows) {
            assertTrue(p[i, i] > 0.0, "$context: P[$i,$i]=${p[i, i]} is not positive")
            for (j in 0 until p.cols) {
                assertEquals(p[i, j], p[j, i], 1e-9, "$context: P not symmetric at ($i,$j)")
            }
        }
    }

    // ---- A. Stationary IMU propagation ----

    @Test
    fun `stationary IMU keeps position and velocity exactly at zero`() {
        val ekf = ErrorStateEkf(NominalState.zero(), modestCovariance())
        val level = Vec3(0.0, 0.0, g) // level, at rest: specific force reads +g on the up axis
        val zeroGyro = Vec3.ZERO
        val dt = 0.01

        repeat(100) { ekf.propagate(level, zeroGyro, dt) } // 1 second

        assertEquals(0.0, ekf.state.position.x, 1e-9)
        assertEquals(0.0, ekf.state.position.y, 1e-9)
        assertEquals(0.0, ekf.state.position.z, 1e-9)
        assertEquals(0.0, ekf.state.velocity.norm(), 1e-9)
    }

    @Test
    fun `stationary propagation keeps covariance symmetric, positive-diagonal, and growing on unobserved blocks`() {
        val ekf = ErrorStateEkf(NominalState.zero(), modestCovariance())
        val level = Vec3(0.0, 0.0, g)
        val initialBiasVariance = ekf.covariance[ErrorStateLayout.ACCEL_BIAS, ErrorStateLayout.ACCEL_BIAS]

        repeat(50) {
            ekf.propagate(level, Vec3.ZERO, 0.01)
            assertSymmetricPositiveDiagonal(ekf.covariance, "step $it")
        }

        // Bias states are never corrected in pure propagation -- their
        // uncertainty must only grow (FR-09's "covariance remains
        // positive-definite every step", checked continuously above, plus
        // this direct check that propagation is not silently shrinking an
        // unobserved state's uncertainty, which would be a bug).
        val finalBiasVariance = ekf.covariance[ErrorStateLayout.ACCEL_BIAS, ErrorStateLayout.ACCEL_BIAS]
        assertTrue(finalBiasVariance > initialBiasVariance, "accel bias variance should grow with no measurement correcting it")
    }

    @Test
    fun `propagation over a long sequence produces no NaN, matching FR-09's 10-minute requirement in miniature`() {
        // A scaled-down version of FR-09's 10-minute acceptance criterion
        // (60000 steps at 100 Hz) -- proves no numerical blow-up over many
        // cycles without a 10-minute test runtime.
        val ekf = ErrorStateEkf(NominalState.zero(), modestCovariance())
        repeat(6000) { ekf.propagate(Vec3(0.0, 0.0, g), Vec3(0.001, -0.001, 0.002), 0.01) }

        assertTrue(ekf.state.position.x.isFinite())
        assertTrue(ekf.state.velocity.norm().isFinite())
        assertSymmetricPositiveDiagonal(ekf.covariance, "after 6000 steps")
    }

    // ---- B. Constant straight-line motion ----

    @Test
    fun `constant forward velocity propagates with no artificial lateral movement`() {
        val initial = NominalState.zero().copy(velocity = Vec3(10.0, 0.0, 0.0))
        val ekf = ErrorStateEkf(initial, modestCovariance())
        // Zero net horizontal specific force (gravity cancels exactly) ->
        // zero acceleration -> constant velocity, per the propagate() math.
        val level = Vec3(0.0, 0.0, g)
        val dt = 0.01
        val steps = 200 // 2 seconds

        repeat(steps) { ekf.propagate(level, Vec3.ZERO, dt) }

        val elapsed = steps * dt
        assertEquals(10.0 * elapsed, ekf.state.position.x, 1e-6)
        assertEquals(0.0, ekf.state.position.y, 1e-9) // the explicit "no artificial lateral movement" check
        assertEquals(0.0, ekf.state.position.z, 1e-9)
        assertEquals(10.0, ekf.state.velocity.x, 1e-9)
        assertEquals(0.0, ekf.state.velocity.y, 1e-9)
    }

    // ---- C. Constant-turn synthetic trajectory ----

    @Test
    fun `constant turn follows the expected circular arc within discretisation tolerance`() {
        // Physics, derived via the transport theorem (not asserted): for a
        // level constant turn at speed v and yaw rate omega, TRUE
        // acceleration in body frame is [0, v*omega, 0] (centripetal,
        // pointing left for a left turn); specific force additionally
        // subtracts body-frame gravity, which for a purely-yawing (no
        // roll/pitch) turn stays [0,0,-g] regardless of heading, giving
        // a_m = [0, v*omega, g].
        val v = 10.0
        val omega = 0.5 // rad/s, gentle left turn
        val radius = v / omega
        val dt = 0.01
        val steps = 200 // 2 s -> 1.0 rad of turn

        val initial = NominalState.zero().copy(velocity = Vec3(v, 0.0, 0.0))
        val ekf = ErrorStateEkf(initial, modestCovariance())
        val accel = Vec3(0.0, v * omega, g)
        val gyro = Vec3(0.0, 0.0, omega)

        repeat(steps) { ekf.propagate(accel, gyro, dt) }

        val elapsed = steps * dt
        val expectedX = radius * sin(omega * elapsed)
        val expectedY = radius * (1.0 - cos(omega * elapsed))

        // Discretisation tolerance, not exactness -- unlike the exact
        // straight-line case, a circular path integrated in small steps
        // accumulates real, bounded numerical error. 2% of the radius is
        // generous enough to be robust, tight enough to catch a wrong-sign
        // or wrong-axis bug (which would miss by 100%, not 2%).
        val tol = radius * 0.02
        assertTrue(abs(ekf.state.position.x - expectedX) < tol, "x: expected ~$expectedX, got ${ekf.state.position.x}")
        assertTrue(abs(ekf.state.position.y - expectedY) < tol, "y: expected ~$expectedY, got ${ekf.state.position.y}")

        // Speed is preserved by a purely centripetal (velocity-perpendicular)
        // acceleration -- an independent physical sanity check, not just
        // position.
        assertEquals(v, ekf.state.velocity.norm(), 0.05)

        // Heading should have advanced by omega*T -- read back via the
        // rotation matrix's own forward-axis direction rather than
        // asserting on quaternion components directly (decouples the test
        // from quaternion sign conventions it doesn't need to know about).
        val forwardNav = ekf.state.orientation.rotate(Vec3(1.0, 0.0, 0.0))
        val expectedHeading = omega * elapsed
        assertEquals(cos(expectedHeading), forwardNav.x, 0.01)
        assertEquals(sin(expectedHeading), forwardNav.y, 0.01)
    }

    // ---- D. Accelerometer bias injection ----

    @Test
    fun `unmodelled accelerometer bias produces quadratic position drift, matching the PRD own arithmetic`() {
        // v3 PRD Section1.2: "error from double integration accumulates
        // as 0.5*b*t^2". This test injects a bias into the SENSOR reading
        // while the filter's own bias STATE stays at zero (no measurement
        // update ever runs to estimate it) -- reproducing the exact
        // failure mode the whole project's thesis exists to avoid, as a
        // concrete regression check on propagate() own math rather than
        // trusting the PRD's arithmetic without checking this
        // implementation actually exhibits it.
        val bias = 0.1 // m/s^2 -- deliberately large versus the PRD's own
        // illustrative 1 mg (~0.0098 m/s^2) so the effect is unambiguous
        // over a short, fast unit test rather than needing minutes of
        // simulated time to separate signal from floating-point noise.
        val ekf = ErrorStateEkf(NominalState.zero(), modestCovariance())
        val dt = 0.01
        val steps = 500 // 5 seconds

        repeat(steps) {
            // Sensor reads level-plus-bias; filter's bias estimate never
            // moves because no correct() call ever runs.
            ekf.propagate(Vec3(bias, 0.0, g), Vec3.ZERO, dt)
        }

        val elapsed = steps * dt
        val expectedDrift = 0.5 * bias * elapsed * elapsed
        assertEquals(expectedDrift, ekf.state.position.x, expectedDrift * 0.01)
        assertEquals(0.0, ekf.state.accelBias.x, 1e-12, "bias STATE must stay at zero with no correction applied")
    }

    @Test
    fun `repeated velocity corrections shrink accelerometer bias covariance and move the bias estimate toward truth`() {
        // Second half of the D test pair: show the filter actually LEARNS
        // the bias given measurements, not just passively tracks it.
        //
        // A SINGLE correction after a short propagation window was tried
        // first and failed: the covariance cross-correlation between
        // velocity-error and bias-error builds up through F_c's
        // off-diagonal coupling (see ErrorStateEkf's own propagate() doc),
        // which is proportional to elapsed time -- over 0.5s that
        // cross-term is genuinely too small for one correction to move
        // the bias estimate meaningfully. That was a test-design mistake,
        // not a filter bug: a real "stopped at a light" scenario applies
        // ZUPT repeatedly over seconds, not once. This test does the same.
        val trueBias = 0.05
        val ekf = ErrorStateEkf(NominalState.zero(), modestCovariance())
        val dt = 0.01
        val biasVarianceBefore = ekf.covariance[ErrorStateLayout.ACCEL_BIAS, ErrorStateLayout.ACCEL_BIAS]

        repeat(500) { // 5 s, biased sensor, corrected every 10 steps (10 Hz ZUPT)
            ekf.propagate(Vec3(trueBias, 0.0, g), Vec3.ZERO, dt)
            if (it % 10 == 9) ekf.correct(ZuptUpdate())
        }

        val biasVarianceAfter = ekf.covariance[ErrorStateLayout.ACCEL_BIAS, ErrorStateLayout.ACCEL_BIAS]
        assertTrue(biasVarianceAfter < biasVarianceBefore, "bias covariance should shrink after repeated correlated corrections")
        // Estimate must be strictly closer to the true bias than staying at
        // zero would be (i.e. genuinely moved toward it, in the right
        // direction, not still parked at the initial zero estimate).
        assertTrue(
            abs(ekf.state.accelBias.x - trueBias) < abs(trueBias),
            "bias estimate (${ekf.state.accelBias.x}) did not move toward the true bias ($trueBias)",
        )
    }

    // ---- Gyroscope bias injection ----
    //
    // Deliberately NOT a re-test of the constant-turn (C) case above under
    // a different name. Worked out before writing this: with the vehicle
    // otherwise stationary (specific force exactly [0,0,g], aligned with
    // the z rotation axis), an erroneous yaw rotation from an unmodelled
    // gyro bias leaves accelNav = R(q)*[0,0,g] + gravity = [0,0,g]+[0,0,-g]
    // = [0,0,0] EXACTLY regardless of the erroneous heading -- rotating a
    // vector already aligned with the rotation axis changes nothing. So a
    // stationary gyro-bias test isolates the PURE ATTITUDE effect (heading
    // drifts, position provably does not), which is a genuinely different,
    // non-redundant check from Test C's demonstration that a nonzero gyro
    // reading combined with a correspondingly-rotating specific force
    // curves the trajectory -- that mechanism is identical whether the
    // rotation is an intentional turn or an unmodelled bias; re-testing it
    // here under a "bias" label would not be checking anything new.

    @Test
    fun `unmodelled gyro bias produces linear heading drift with no accompanying position drift`() {
        // b and T chosen to match v3 PRD Section1.2's own worked example
        // (0.1 deg/s residual yaw bias) exactly, so the result is directly
        // comparable to the PRD's own number, not an arbitrary magnitude.
        val bias = 0.1 * Math.PI / 180.0 // 0.1 deg/s in rad/s, per PRD Section1.2
        val ekf = ErrorStateEkf(NominalState.zero(), modestCovariance())
        val dt = 0.01
        val steps = 6000 // 60 s, matching the PRD's own example duration

        repeat(steps) { ekf.propagate(Vec3(0.0, 0.0, g), Vec3(0.0, 0.0, bias), dt) }

        val elapsed = steps * dt
        val expectedHeading = bias * elapsed // ~0.1047 rad, ~6.0 degrees
        val forwardNav = ekf.state.orientation.rotate(Vec3(1.0, 0.0, 0.0))
        assertEquals(cos(expectedHeading), forwardNav.x, 1e-6)
        assertEquals(sin(expectedHeading), forwardNav.y, 1e-6)

        // The point of isolating this from Test C: with zero net specific
        // force imbalance, the erroneous heading has nothing to misdirect
        // -- position must stay at EXACTLY zero despite attitude having
        // drifted by several degrees. (Cross-track position error, per the
        // PRD's own formula, is what happens when this same heading drift
        // ALSO redirects a real forward acceleration -- that is Test C's
        // mechanism, not repeated here.)
        assertEquals(0.0, ekf.state.position.x, 1e-9)
        assertEquals(0.0, ekf.state.position.y, 1e-9)
        assertEquals(0.0, ekf.state.velocity.norm(), 1e-9)
        assertEquals(0.0, ekf.state.gyroBias.z, 1e-12, "bias STATE must stay at zero with no correction applied")
    }

    /** Synthetic, test-only direct attitude-error observation -- the exact
     *  analogue of ErrorStateEkfCorrectionTest's DirectPositionXObservation,
     *  for the theta block instead of position. Honestly not a real
     *  sensor (no phone sensor observes attitude error directly); it
     *  exists to isolate and demonstrate the delta-theta <-> delta-b_g
     *  Kalman coupling on its own, the same way that test isolates
     *  delta-p <-> nothing and VelocityUpdate exercises delta-v <-> delta-b_a. */
    private class DirectYawErrorObservation(private val observedRadians: Double, private val varianceValue: Double) : MeasurementModel {
        override val name = "TestDirectYawError"
        override val dimension = 1
        override fun predicted(state: NominalState): Vec {
            // First-order proxy for the z-component of attitude error: for
            // a pure-yaw rotation from identity, atan2(forward.y, forward.x)
            // IS the yaw angle exactly, not just to first order.
            val forward = state.orientation.rotate(Vec3(1.0, 0.0, 0.0))
            return Vec.of(kotlin.math.atan2(forward.y, forward.x))
        }
        override fun actual(): Vec = Vec.of(observedRadians)
        override fun jacobian(state: NominalState): Mat {
            val h = Mat.zeros(1, ErrorStateLayout.DIM)
            h[0, ErrorStateLayout.THETA + 2] = 1.0 // selects delta-theta_z directly
            return h
        }
        override fun noise(): Mat = Mat(1, 1, doubleArrayOf(varianceValue))
    }

    @Test
    fun `repeated direct attitude corrections shrink gyroscope bias covariance and move the estimate toward truth`() {
        // Structural mirror of the accelerometer-bias-learning test above:
        // propagate with an unmodelled gyro bias, interleave repeated
        // corrections, confirm the bias ESTIMATE moves toward truth and
        // its covariance shrinks. Uses DirectYawErrorObservation rather
        // than a real measurement model because none of NHC/ZUPT/
        // VelocityUpdate observe attitude at all (they are all body-
        // velocity observations) -- this deliberately tests the filter's
        // generic delta-theta<->delta-b_g coupling in isolation, the same
        // way Test E's generic-update test isolates the correction
        // machinery from any specific physical model.
        val trueBias = 0.05 // rad/s
        val ekf = ErrorStateEkf(NominalState.zero(), modestCovariance())
        val dt = 0.01
        val biasVarianceBefore = ekf.covariance[ErrorStateLayout.GYRO_BIAS + 2, ErrorStateLayout.GYRO_BIAS + 2]

        repeat(500) { // 5 s, same window as the proven-sufficient accel-bias case
            ekf.propagate(Vec3(0.0, 0.0, g), Vec3(0.0, 0.0, trueBias), dt)
            if (it % 10 == 9) ekf.correct(DirectYawErrorObservation(observedRadians = 0.0, varianceValue = 0.01))
        }

        val biasVarianceAfter = ekf.covariance[ErrorStateLayout.GYRO_BIAS + 2, ErrorStateLayout.GYRO_BIAS + 2]
        assertTrue(biasVarianceAfter < biasVarianceBefore, "gyro bias covariance should shrink after repeated correlated corrections")
        assertTrue(
            abs(ekf.state.gyroBias.z - trueBias) < abs(trueBias),
            "gyro bias estimate (${ekf.state.gyroBias.z}) did not move toward the true bias ($trueBias)",
        )
    }
}
