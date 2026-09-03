"""Per-sequence usability scoring for the synchronised subset.

Measured on 2026-09-03 (ml/docs/IO-VNBD-verification.md §8): the phone in
IO-VNBD's synchronised drives is mounted inconsistently. Some sequences (S1,
S3c) carry vehicle motion cleanly in the phone IMU; many Driver-E sequences
have the phone loose enough that its IMU is dominated by non-vehicle motion and
carries almost no speed information; Driver D (Y1) looks decoupled entirely.

Training the mean/variance heads on sequences where the phone IMU is pure noise
w.r.t. the label teaches nothing and corrupts calibration, so every sequence
gets a usability label and the split protocol + label_sigma use it. This is NOT
throwing data away arbitrarily — it is the same signal the deployed system
gets from Kamal's AlignmentService quality output, applied at training time.

Scores (all at the shared 10 Hz row clock, zero lag — time alignment is already
good, confirmed via phone-GPS-speed vs vehicle-speed):

  vib_speed_corr : corr( rolling-std(|phone accel|, 1 s), vehicle speed ).
                   The thesis signal — does vibration texture track speed.
  lsq_yaw_r2     : R^2 of  vehicle_yaw_rate ~ [phone gyro x/y/z] least squares.
                   Recovers the (unknown) mount rotation's yaw projection;
                   high => phone rigidly carries vehicle rotation.
  sync_speed_corr: corr( phone GPS speed, vehicle speed ), zero lag. Low =>
                   the pair may be mis-synchronised, not just noisy.
  move_fraction  : fraction of samples with vehicle speed > 1 m/s.
  turn_fraction  : fraction of samples with |vehicle yaw rate| > 0.05 rad/s.

usability:
  "use"  vib_speed_corr >= 0.35  OR  lsq_yaw_r2 >= 0.5
  "weak" vib_speed_corr >= 0.20  OR  lsq_yaw_r2 >= 0.30      (train with inflated sigma)
  "drop" otherwise, or move_fraction < 0.05 (vehicle basically parked),
         or sync_speed_corr < 0.3 (suspect sync)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract import SAMPLE_RATE_HZ

_ROLL = SAMPLE_RATE_HZ  # 1 s window


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    if not m.any():
        return np.zeros_like(x)
    x = np.where(m, x, x[m].mean())
    s = x.std()
    return (x - x.mean()) / s if s else x * 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _z(a), _z(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_sequence(df: pd.DataFrame) -> dict:
    n = len(df)
    veh_speed = df["veh_speed_mps"].to_numpy()
    yaw = df["veh_yaw_rate_radps"].to_numpy()

    a_mag = np.sqrt(
        df["phone_accel_x_mps2"] ** 2
        + df["phone_accel_y_mps2"] ** 2
        + df["phone_accel_z_mps2"] ** 2
    ).to_numpy()
    a_vib = pd.Series(a_mag).rolling(_ROLL, center=True).std().to_numpy()
    vib_speed_corr = _corr(a_vib, veh_speed)

    G = np.c_[
        df["phone_gyro_yaw_radps"].to_numpy(),
        df["phone_gyro_pitch_radps"].to_numpy(),
        df["phone_gyro_roll_radps"].to_numpy(),
        np.ones(n),
    ]
    mask = np.all(np.isfinite(G), axis=1) & np.isfinite(yaw)
    lsq_yaw_r2 = float("nan")
    if mask.sum() > 200 and yaw[mask].std() > 1e-3:
        beta, *_ = np.linalg.lstsq(G[mask], yaw[mask], rcond=None)
        pred = G[mask] @ beta
        ss_res = float(np.sum((yaw[mask] - pred) ** 2))
        ss_tot = float(np.sum((yaw[mask] - yaw[mask].mean()) ** 2))
        lsq_yaw_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    sync_speed_corr = _corr(df["phone_gps_speed_mps"].to_numpy(), veh_speed)
    move_fraction = float(np.mean(veh_speed > 1.0))
    turn_fraction = float(np.mean(np.abs(yaw) > 0.05))

    v = np.nan_to_num(vib_speed_corr)
    r = np.nan_to_num(lsq_yaw_r2)
    s = np.nan_to_num(sync_speed_corr, nan=1.0)  # nan sync corr (parked) handled by move check
    if move_fraction < 0.05 or s < 0.30:
        usability = "drop"
    elif v >= 0.35 or r >= 0.50:
        usability = "use"
    elif v >= 0.20 or r >= 0.30:
        usability = "weak"
    else:
        usability = "drop"

    return {
        "vib_speed_corr": _round(vib_speed_corr),
        "lsq_yaw_r2": _round(lsq_yaw_r2),
        "sync_speed_corr": _round(sync_speed_corr),
        "move_fraction": round(move_fraction, 3),
        "turn_fraction": round(turn_fraction, 3),
        "usability": usability,
    }


def _round(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)
