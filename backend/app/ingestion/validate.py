"""Server-side validation + bounds + consent re-check for /v1/telemetry/labels
(PRD §5.2 telemetry ingestion, FR-23 server-side 403 consent re-check).

Position-stripped by construction — the LabelPair model has no lat/lon field.
What we still enforce:
  * contract_version of each window is one this server can interpret,
  * the IMU window is the right shape (WINDOW_SIZE_SAMPLES x NUM_FEATURES),
  * physical plausibility (same FR-24 bounds the model output uses):
    displacement over window_duration_s implies a mean speed in [0, 100] m/s,
    and no |accel| / |gyro| far outside a phone-in-a-car range,
  * gyro is rad/s not deg/s (contracts/units.md ~57x guard).
"""
from __future__ import annotations

import numpy as np

MAX_MEAN_SPEED_MPS = 100.0
GYRO_ABS_MAX_RADPS = 10.0
ACCEL_ABS_MAX_MPS2 = 100.0


def validate_pair(pair, *, supported_contract_prefixes=("1.",)) -> str | None:
    """Return a rejection reason string, or None if the pair is acceptable."""
    if not any(pair.contract_version.startswith(p) for p in supported_contract_prefixes):
        return f"unsupported contract_version {pair.contract_version}"

    w = np.asarray(pair.imu_window, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 6:
        return f"imu_window shape {w.shape} != [T, 6]"
    if not np.all(np.isfinite(w)):
        return "imu_window has non-finite values"

    if pair.window_duration_s <= 0:
        return "window_duration_s must be > 0"
    mean_speed = pair.displacement_m / pair.window_duration_s
    if not (0.0 <= mean_speed <= MAX_MEAN_SPEED_MPS):
        return f"implied mean speed {mean_speed:.1f} m/s outside [0, {MAX_MEAN_SPEED_MPS}]"

    accel, gyro = w[:, 0:3], w[:, 3:6]
    if np.abs(accel).max() > ACCEL_ABS_MAX_MPS2:
        return "accel magnitude implausible (unit error?)"
    if np.percentile(np.abs(gyro), 99.5) > GYRO_ABS_MAX_RADPS:
        return "gyro looks like deg/s not rad/s (contracts/units.md)"
    return None


def validate_batch(pairs, *, consent_ok: bool) -> tuple[int, int, list[str]]:
    if not consent_ok:
        return 0, len(pairs), ["consent not on file for this device (FR-23)"]
    accepted = 0
    reasons: list[str] = []
    for p in pairs:
        r = validate_pair(p)
        if r is None:
            accepted += 1
        else:
            reasons.append(r)
    return accepted, len(pairs) - accepted, reasons
