"""Per-window displacement labels and their uncertainty (label_sigma_m).

Target (PRD §6.4): scalar forward displacement over the 2.0 s window, in metres,
from the vehicle's four wheel-speed channels (rad/s) integrated and scaled by
wheel radius, cross-checked against CAN and VBOX speed. The disagreement is not
swept under the rug — it becomes the per-sample label uncertainty the variance
head (Head B) trains against. A model told its labels are perfect learns to be
overconfident; that is the exact failure FR-08's calibration test catches.

--- wheel radius (PRD §6.4: "derive by regression, do not look it up") ---
One instrumented vehicle => one radius. Fit through the origin on straight
(|yaw rate| < 0.03 rad/s), moving (VBOX speed > 5 m/s), GNSS-clean stretches:
    r  =  Σ(v_vbox · ω_mean)  /  Σ(ω_mean²)
where ω_mean is the mean of the 4 wheel angular rates. Also fit per-sequence and
report the spread (tyre pressure / temperature drift across drives).

--- label_sigma_m: FOUR independent contributions, combined in quadrature ---
  σ_wheelcan : within-window disagreement between wheel-integrated distance and
               CAN "indicated vehicle speed" integrated distance. Catches wheel
               slip, a locked/spinning wheel, gear-change transients.
  σ_gnss     : the RMS residual of (r·ω_mean − v_vbox) over this sequence's OWN
               clean straight stretches, × window duration. This is the floor:
               even with perfect wheels, VBOX GPS speed is metre-class, so the
               "true" displacement we regressed against is itself uncertain.
               REQUIRED — without it the variance head trains against an
               artificially tight label and calibrates optimistically.
  σ_sync     : a phone/vehicle timing offset τ mislabels a MEAN-SPEED window by
               ~ τ · |Δspeed across the window| (NOT τ · speed — the mean speed
               of a smooth 2 s window barely moves under a small shift; only a
               hard accel/brake window is genuinely more uncertain). Measured
               zero-lag phone↔vehicle sync is tight (sync_speed_corr 0.75-0.97),
               so τ ≈ 0.3 s, plus a per-sequence constant scaled by
               (1 - sync_speed_corr) for the residual whole-sequence offset.
               [earlier revision used τ · speed, which swamped the label at ~3-7
               m/s and starved the mean head — see ml/docs/training-notes.md]
  σ_mount    : sequence-usability multiplier folded in as a relative term
               (use ×1.0, weak ×1.6, drop ×3.0) on the combined above — a
               loosely-mounted phone's window is a worse training example and
               the head should be told so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import SAMPLE_RATE_HZ

DT_S = 1.0 / SAMPLE_RATE_HZ
STRAIGHT_YAW_RADPS = 0.03
CLEAN_MIN_SPEED_MPS = 5.0
CLEAN_MIN_SATS = 4
SYNC_UNCERTAINTY_S = 0.3
SYNC_SEQ_OFFSET_SCALE_M = 2.0        # per-seq constant = (1 - sync_speed_corr) * this
_USABILITY_MULT = {"use": 1.0, "weak": 1.4, "drop": 3.0, "excluded": 3.0}
WHEEL_COLS = ["veh_wheel_fl_radps", "veh_wheel_fr_radps",
              "veh_wheel_rl_radps", "veh_wheel_rr_radps"]


@dataclass
class WheelRadiusFit:
    radius_m: float
    per_sequence_m: dict[str, float]
    n_samples: int
    spread_m: float          # std of per-sequence radii


def _clean_mask(df) -> np.ndarray:
    return (
        (np.abs(df["veh_yaw_rate_radps"].to_numpy()) < STRAIGHT_YAW_RADPS)
        & (df["veh_speed_mps"].to_numpy() > CLEAN_MIN_SPEED_MPS)
        & (df["veh_gps_sats"].to_numpy() >= CLEAN_MIN_SATS)
        & np.all(np.isfinite(df[WHEEL_COLS].to_numpy()), axis=1)
        & np.isfinite(df["veh_speed_mps"].to_numpy())
    )


def fit_wheel_radius(sequences) -> WheelRadiusFit:
    """`sequences` = the TRAIN split only (never fit calibration/scale on test)."""
    num = den = 0.0
    n = 0
    per_seq: dict[str, float] = {}
    for seq in sequences:
        df = seq.df
        mask = _clean_mask(df)
        if mask.sum() < 100:
            continue
        omega = df[WHEEL_COLS].to_numpy()[mask].mean(axis=1)
        v = df["veh_speed_mps"].to_numpy()[mask]
        num += float(np.sum(v * omega))
        den += float(np.sum(omega * omega))
        n += int(mask.sum())
        per_seq[seq.seq_id] = float(np.sum(v * omega) / np.sum(omega * omega))
    if den == 0:
        raise ValueError("no clean straight stretches to fit wheel radius on")
    radius = num / den
    spread = float(np.std(list(per_seq.values()))) if per_seq else 0.0
    return WheelRadiusFit(radius_m=radius, per_sequence_m=per_seq,
                          n_samples=n, spread_m=spread)


def _seq_gnss_speed_rmse(df, radius_m: float) -> float:
    mask = _clean_mask(df)
    if mask.sum() < 50:
        return 0.20  # default metre-class floor if a sequence has no clean run
    omega = df[WHEEL_COLS].to_numpy()[mask].mean(axis=1)
    resid = radius_m * omega - df["veh_speed_mps"].to_numpy()[mask]
    return float(np.sqrt(np.mean(resid ** 2)))


@dataclass
class WindowLabel:
    displacement_m: float          # target for Head A (see contract note in frame doc)
    mean_speed_mps: float          # displacement_m / window_duration_s
    label_sigma_m: float
    parts: dict                    # the four sigma contributions, for auditing


class SequenceLabeller:
    """Produces a WindowLabel for any [start, stop) row range of one sequence."""

    def __init__(self, seq, radius_m: float):
        self.seq = seq
        self.r = radius_m
        df = seq.df
        self.omega_mean = df[WHEEL_COLS].to_numpy().mean(axis=1)
        self.indicated = df["veh_indicated_speed_mps"].to_numpy()
        self.vbox = df["veh_speed_mps"].to_numpy()
        self.seq_gnss_rmse = _seq_gnss_speed_rmse(df, radius_m)
        self.usability = seq.meta.get("usability", "weak")
        sc = seq.meta.get("sync_speed_corr")
        self.sync_offset_m = SYNC_SEQ_OFFSET_SCALE_M * (1.0 - (sc if sc is not None else 0.85))

    def label(self, start: int, stop: int) -> WindowLabel:
        sl = slice(start, stop)
        dur = (stop - start) * DT_S
        wheel_speed = self.r * self.omega_mean[sl]
        disp = float(np.sum(wheel_speed) * DT_S)          # left-Riemann, 10 Hz
        mean_speed = disp / dur if dur > 0 else 0.0

        # σ_wheelcan : wheel vs CAN indicated speed, integrated over the window
        can_speed = self.indicated[sl]
        s_wheelcan = float(np.nansum(np.abs(wheel_speed - can_speed)) * DT_S)
        s_wheelcan = 0.5 * s_wheelcan  # mean, not sum, of the abs disagreement path

        # σ_gnss : sequence clean-stretch speed RMSE, propagated over the window
        s_gnss = self.seq_gnss_rmse * dur

        # σ_sync : timing-offset error ~ tau * |Δspeed over window|, plus a
        # per-sequence constant for a residual whole-sequence offset.
        dspeed = float(abs(wheel_speed[-1] - wheel_speed[0])) if len(wheel_speed) else 0.0
        s_sync = float(np.hypot(SYNC_UNCERTAINTY_S * dspeed, self.sync_offset_m))

        base = float(np.sqrt(s_wheelcan ** 2 + s_gnss ** 2 + s_sync ** 2))
        sigma = base * _USABILITY_MULT.get(self.usability, 1.6)
        sigma = max(sigma, 0.05)  # never claim sub-5-cm certainty on a metre-class GT

        return WindowLabel(
            displacement_m=disp,
            mean_speed_mps=mean_speed,
            label_sigma_m=sigma,
            parts={
                "sigma_wheelcan_m": round(s_wheelcan, 4),
                "sigma_gnss_m": round(s_gnss, 4),
                "sigma_sync_m": round(s_sync, 4),
                "sigma_sync_seq_offset_m": round(self.sync_offset_m, 4),
                "dspeed_over_window_mps": round(dspeed, 4),
                "usability_mult": _USABILITY_MULT.get(self.usability, 1.4),
                "seq_gnss_speed_rmse_mps": round(self.seq_gnss_rmse, 4),
            },
        )
