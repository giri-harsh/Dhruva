package org.anchor.fusion

/**
 * Wires ChiSquareGate into the actual measurement-update path (FR-27):
 * compute a measurement's innovation and innovation covariance, gate it,
 * and only call ErrorStateEkf.correct() if the gate accepts. A rejected
 * measurement is reported to the caller, never silently dropped and never
 * applied -- this is GnssPositionUpdate's own doc comment's prescription
 * ("the orchestrator that eventually calls this should run
 * ChiSquareGate.test() ... BEFORE calling ErrorStateEkf.correct()"), made
 * real.
 *
 * Generic over MeasurementModel, not GNSS-specific -- the same path any
 * future measurement (ANCHOR-Net's VelocityUpdate included) would go
 * through, composition only, no new per-model gating logic needed.
 *
 * Innovation/S are necessarily recomputed here rather than read out of
 * ErrorStateEkf.correct(): that method commits to a correction
 * unconditionally once called (Week 2's own, reviewed design, not to be
 * changed this week), so there is no way to ask it "what would the
 * innovation be" without also applying it. The two lines below duplicate
 * only the standard nu=z-h(x), S=HPH^T+R formula ChiSquareGate.test()
 * itself already takes as input -- not any ESKF-internal derivation.
 */
object GatedMeasurementUpdate {
    fun apply(
        ekf: ErrorStateEkf,
        measurement: MeasurementModel,
        confidence: ChiSquareGate.Confidence = ChiSquareGate.Confidence.P95,
    ): ChiSquareGate.Result {
        val h = measurement.jacobian(ekf.state)
        val r = measurement.noise()
        val innovation = measurement.actual() - measurement.predicted(ekf.state)
        val innovationCovariance = (h * ekf.covariance) * h.transpose() + r

        val result = ChiSquareGate.test(innovation, innovationCovariance, confidence)
        if (result.accepted) {
            ekf.correct(measurement)
        }
        return result
    }
}
