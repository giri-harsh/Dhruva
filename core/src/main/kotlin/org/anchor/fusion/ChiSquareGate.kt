package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec

/**
 * FR-27: gates a measurement by testing its normalised innovation
 * nu^T S^-1 nu against a chi-square critical value for the measurement
 * dimension. Generic -- usable by any MeasurementModel, not GNSS-
 * specific, despite FR-27's own wording being framed around GNSS; the
 * PS's "innovation gate" concept is exactly this statistical test,
 * independent of which sensor produced the innovation.
 *
 * "This gate detects DISCONTINUOUS disagreement between a measurement and
 * what the filter physically believes -- v3 PRD Section15.1 T3's own
 * words: 'detects discontinuous spoofing and multipath', never
 * 'spoof-proof'. A slow, patient drift inside the gate the whole way is
 * NOT caught by this test; FR-31 (Harshit's integrity bench) measures
 * exactly where that boundary sits. Stated here, not just in the GNSS
 * threat model, because this class is the actual mechanism that
 * boundary describes.
 */
object ChiSquareGate {

    enum class Confidence { P95, P99 }

    /** Standard chi-square critical values, upper-tail (P(X > value) =
     *  1-confidence), for the degrees of freedom this project actually
     *  uses (1: VelocityUpdate; 2: NHC, a 2D GNSS position/velocity
     *  component; 3: ZUPT, a 3D GNSS fix). Sourced from a standard
     *  chi-square distribution table -- not computed, not approximated,
     *  so there is nothing here to get subtly wrong the way a numerical
     *  inverse-CDF solver could. */
    private val CRITICAL_VALUES: Map<Pair<Int, Confidence>, Double> = mapOf(
        (1 to Confidence.P95) to 3.841,
        (2 to Confidence.P95) to 5.991,
        (3 to Confidence.P95) to 7.815,
        (4 to Confidence.P95) to 9.488,
        (5 to Confidence.P95) to 11.070,
        (6 to Confidence.P95) to 12.592,
        (1 to Confidence.P99) to 6.635,
        (2 to Confidence.P99) to 9.210,
        (3 to Confidence.P99) to 11.345,
        (4 to Confidence.P99) to 13.277,
        (5 to Confidence.P99) to 15.086,
        (6 to Confidence.P99) to 16.812,
    )

    data class Result(val statistic: Double, val threshold: Double, val accepted: Boolean)

    /** innovation: z - h(x). innovationCovariance: S = HPH^T + R. */
    fun test(innovation: Vec, innovationCovariance: Mat, confidence: Confidence = Confidence.P95): Result {
        require(innovationCovariance.rows == innovation.size && innovationCovariance.cols == innovation.size) {
            "innovationCovariance must be ${innovation.size}x${innovation.size} to match the innovation dimension"
        }
        val threshold = CRITICAL_VALUES[innovation.size to confidence]
            ?: error(
                "ChiSquareGate has no tabulated critical value for dimension ${innovation.size} at $confidence " +
                    "-- add the correct value from a chi-square table rather than approximating it.",
            )
        // nu^T S^-1 nu
        val sInv = innovationCovariance.inverse()
        val statistic = innovation.dot(sInv * innovation)
        return Result(statistic, threshold, accepted = statistic <= threshold)
    }
}
