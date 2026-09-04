package org.anchor.orchestrator

import org.anchor.fusion.ChiSquareGate
import org.anchor.fusion.ProcessNoiseConfig

/**
 * Physically meaningful, documented thresholds for EngineOrchestrator's
 * own scheduling decisions -- kept separate from ProcessNoiseConfig
 * (filter-internal noise tuning) because these are orchestration-level
 * choices (WHEN calibration is trusted, WHEN a GNSS fix is gated) rather
 * than anything ErrorStateEkf itself needs to know about.
 */
data class EngineConfig(
    val processNoise: ProcessNoiseConfig = ProcessNoiseConfig(),
    /** FR-04's own stated minimum: "as little as 3s of stationary or
     *  steady motion" is enough for gravity-based roll/pitch levelling.
     *  Not a newly-invented number -- FR-04's own figure, converted to
     *  nanoseconds because SensorEvent.timestampNanos is this engine's
     *  only clock (sample RATE varies by device/source, so a time window
     *  is used rather than a sample count). */
    val calibrationDurationNanos: Long = 3_000_000_000L,
    /** Matches StationarityDetector's own default -- not re-declared as a
     *  second, possibly-diverging default; see StationarityDetector.kt
     *  for the [VERIFY]-flagged variance thresholds themselves. */
    val stationarityWindowSize: Int = 20,
    val gnssGateConfidence: ChiSquareGate.Confidence = ChiSquareGate.Confidence.P95,
) {
    init {
        require(calibrationDurationNanos > 0) { "calibrationDurationNanos must be positive" }
        require(stationarityWindowSize >= 2) { "stationarityWindowSize must be >= 2 (StationarityDetector's own requirement)" }
    }
}
