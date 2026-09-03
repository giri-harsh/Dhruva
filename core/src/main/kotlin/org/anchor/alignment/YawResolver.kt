package org.anchor.alignment

import org.anchor.math.Mat3
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * FLAGGED DIVERGENCE, not a silent substitution -- read this before
 * touching yaw resolution.
 *
 * ml/anchor/data/features.py resolves yaw by correlating levelled
 * longitudinal accel against veh_long_accel_mps2, which is CAN vehicle-bus
 * data present in Harshit training set (IO-VNBD). That signal exists in
 * NEITHER of the two environments this track ships to: it is not a
 * channel Android exposes to an app, and it is not a column in the frozen
 * contracts/replay_csv/schema.json either (checked directly against the
 * committed schema -- there is no veh_long_accel column there).
 *
 * FR-05 own acceptance criterion already names the correct on-device
 * substitute: "the estimate agrees with GNSS course-over-ground... while
 * GNSS is available." A LongitudinalReference is exactly that substitute,
 * injected rather than hardcoded, so the SEARCH ALGORITHM below (the 72-
 * point grid, the correlation scoring, the two-part fallback gate) is a
 * faithful, bit-comparable port of _best_yaw, while the ONE input that
 * cannot possibly exist on a phone is isolated behind an interface instead
 * of being quietly reinvented inside the algorithm.
 */
fun interface LongitudinalReference {
    /** m/s^2, vehicle-forward-axis-projected longitudinal acceleration,
     *  or null where no reference is available for that sample (e.g. no
     *  GNSS fix at that moment) -- callers must filter nulls exactly as
     *  the Python reference filters non-finite values via np.isfinite. */
    fun at(index: Int): Double?
}

object YawResolver {

    /** Direct port of _yaw_rotation: standard right-handed rotation about
     *  the (already-levelled) vehicle z-axis by theta radians. */
    fun yawRotation(theta: Double): Mat3 {
        val c = cos(theta)
        val s = sin(theta)
        return Mat3(
            c, -s, 0.0,
            s, c, 0.0,
            0.0, 0.0, 1.0,
        )
    }

    /**
     * Direct port of _best_yaw's search, generalised only in WHERE the
     * reference signal comes from (see LongitudinalReference doc above).
     * Every numeric detail matches the Python source: 72 candidate angles
     * spanning [-pi, pi) at exactly pi/36 (5 degree) steps, a minimum of
     * 200 valid paired samples, a minimum reference standard deviation of
     * 1e-3 below which no correction is attempted (returns 0.0, i.e.
     * identity yaw -- matching features.py fallback exactly), and a
     * Pearson-correlation score per candidate angle.
     *
     * accelLevelXY: the horizontal-plane (x, y) components of accel AFTER
     * GravityAlign has already been applied -- i.e. accel_lvl[:, :2] in
     * the Python source, passed in pre-sliced rather than sliced here, so
     * this function has no 3-vector dependency at all.
     */
    fun bestYaw(accelLevelXY: List<Pair<Double, Double>>, reference: LongitudinalReference): Double {
        require(accelLevelXY.isNotEmpty()) { "bestYaw requires at least one sample" }

        val pairedAx = ArrayList<Double>(accelLevelXY.size)
        val pairedAy = ArrayList<Double>(accelLevelXY.size)
        val pairedY = ArrayList<Double>(accelLevelXY.size)
        for (i in accelLevelXY.indices) {
            val (ax, ay) = accelLevelXY[i]
            val y = reference.at(i)
            if (ax.isFinite() && ay.isFinite() && y != null && y.isFinite()) {
                pairedAx.add(ax)
                pairedAy.add(ay)
                pairedY.add(y)
            }
        }

        if (pairedY.size < 200) return 0.0
        val yArray = pairedY.toDoubleArray()
        val yStd = populationStd(yArray)
        if (yStd < 1e-3) return 0.0

        val yMean = yArray.average()
        val yCentered = DoubleArray(yArray.size) { yArray[it] - yMean }
        val yCenteredStd = populationStd(yCentered) // translation-invariant, equals yStd; kept
        // as a separate variable to mirror the Python source computing
        // std(y) inside the loop on the already-centered array, not to
        // imply a different value -- see the doc comment on why this is
        // deliberately redundant rather than simplified away.

        var bestTheta = 0.0
        var bestScore = Double.NEGATIVE_INFINITY

        val step = (2.0 * PI) / 72.0
        for (i in 0 until 72) {
            val theta = -PI + i * step
            val cosT = cos(theta)
            val sinT = sin(theta)

            val fwd = DoubleArray(pairedAx.size) { pairedAx[it] * cosT + pairedAy[it] * sinT }
            val fwdMean = fwd.average()
            val fwdCentered = DoubleArray(fwd.size) { fwd[it] - fwdMean }
            val fwdStd = populationStd(fwdCentered)

            val denom = fwdStd * yCenteredStd
            val score = if (denom > 0.0) {
                var sumProduct = 0.0
                for (k in fwdCentered.indices) sumProduct += fwdCentered[k] * yCentered[k]
                (sumProduct / fwdCentered.size) / denom
            } else {
                0.0
            }

            if (score > bestScore) {
                bestScore = score
                bestTheta = theta
            }
        }

        return bestTheta
    }

    private fun populationStd(values: DoubleArray): Double {
        if (values.isEmpty()) return 0.0
        val mean = values.average()
        val variance = values.sumOf { (it - mean) * (it - mean) } / values.size
        return sqrt(variance)
    }
}
