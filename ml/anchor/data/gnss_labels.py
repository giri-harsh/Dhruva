"""Weak per-window speed labels from phone GNSS, for the Stage-1 pre-train.

The phone logs GPS Doppler speed at ~1 Hz, forward-filled onto the 10 Hz grid.
Over a 2 s window the mean of that is a usable but noisy speed target:
  * Doppler speed noise ~0.5-1.5 m/s depending on `gps_accuracy_m`,
  * ~0.5-1 s latency on acceleration,
  * forward-fill staleness.
So `label_sigma` here is ~1.5-3 m/s (vs ~0.2 for the synchronised wheel labels)
and windows with poor GNSS are dropped entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import SAMPLE_RATE_HZ

DT_S = 1.0 / SAMPLE_RATE_HZ
MAX_ACCURACY_M = 20.0        # drop windows whose GNSS accuracy is worse
BASE_SPEED_SIGMA_MPS = 1.2
ACCURACY_SIGMA_COEF = 0.15   # + this * gps_accuracy_m
LATENCY_SIGMA_COEF = 0.10    # + this * |Δspeed across window| (lag term)


@dataclass
class GnssWindowLabel:
    mean_speed_mps: float
    label_sigma_mps: float
    ok: bool
    reason: str = ""


class GnssSpeedLabeller:
    def __init__(self, seq):
        d = seq.df
        self.speed = d["phone_gps_speed_mps"].to_numpy() if "phone_gps_speed_mps" in d \
            else d["gps_speed_mps"].to_numpy()
        acc = "phone_gps_accuracy_m" if "phone_gps_accuracy_m" in d else "gps_accuracy_m"
        self.acc = d[acc].to_numpy()

    def label(self, start: int, stop: int) -> GnssWindowLabel:
        sp = self.speed[start:stop]
        ac = self.acc[start:stop]
        if not np.all(np.isfinite(sp)):
            return GnssWindowLabel(0.0, 0.0, False, "non-finite gps speed")
        acc_med = float(np.nanmedian(ac)) if np.any(np.isfinite(ac)) else 99.0
        if acc_med > MAX_ACCURACY_M:
            return GnssWindowLabel(0.0, 0.0, False, f"gps accuracy {acc_med:.0f} m")
        n_distinct = len(np.unique(np.round(sp, 3)))
        if n_distinct < 2 and sp.mean() > 1.0:
            # a whole window of a single forward-filled value while moving:
            # the GNSS is stale — usable but extra-uncertain, not dropped
            stale = True
        else:
            stale = False

        mean_speed = float(np.mean(sp))
        dspeed = float(abs(sp[-1] - sp[0]))
        sigma = (BASE_SPEED_SIGMA_MPS
                 + ACCURACY_SIGMA_COEF * acc_med
                 + LATENCY_SIGMA_COEF * dspeed
                 + (1.0 if stale else 0.0))
        return GnssWindowLabel(mean_speed, sigma, True)
