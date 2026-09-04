"""Label an outage window by its vehicle dynamics (PRD §6.7 — "start points cover
the scenario mix ... report the per-scenario breakdown, averages hide the
roundabout").

Rule-based from the CAN/VBOX channels over the window (ground truth is available
for a held-out sequence — this is scoring metadata, not a model input):

  stop_start       speed dips below 1 m/s at some point
  hard_braking     min longitudinal accel < -2.5 m/s^2
  sharp_cornering  peak |yaw rate| > 0.35 rad/s (~20 deg/s)
  roundabout       sustained moderate yaw (>0.12 rad/s for >3 s) with a speed
                   trough between two higher-speed stretches
  motorway_cruise  mean speed > 22 m/s and yaw p95 < 0.05 rad/s
  urban            everything else (moderate speed, intermittent manoeuvres)

Priority order = the list above (a window that both brakes hard and stops is
`stop_start` only if it also stops; otherwise `hard_braking`). Exactly one label.
"""
from __future__ import annotations

import numpy as np

from ..contract import SAMPLE_RATE_HZ

_SCENARIOS = ("stop_start", "hard_braking", "sharp_cornering", "roundabout",
              "motorway_cruise", "urban")


def classify_window(seq, start: int, stop: int) -> str:
    d = seq.df
    speed = d["veh_speed_mps"].to_numpy()[start:stop]
    yaw = np.abs(d["veh_yaw_rate_radps"].to_numpy()[start:stop])
    long_acc = d["veh_long_accel_mps2"].to_numpy()[start:stop]

    if np.nanmin(speed) < 1.0:
        return "stop_start"
    if np.nanmin(long_acc) < -2.5:
        return "hard_braking"
    if np.nanmax(yaw) > 0.35:
        return "sharp_cornering"

    turning = yaw > 0.12
    if turning.sum() > 3 * SAMPLE_RATE_HZ:
        mid = len(speed) // 2
        if (np.nanmean(speed[:mid // 2 + 1]) - np.nanmin(speed) > 3.0
                and np.nanmean(speed[-(mid // 2 + 1):]) - np.nanmin(speed) > 3.0):
            return "roundabout"

    if np.nanmean(speed) > 22.0 and np.nanpercentile(yaw, 95) < 0.05:
        return "motorway_cruise"
    return "urban"


def scenario_mix(labels) -> dict:
    from collections import Counter
    c = Counter(labels)
    return {s: c.get(s, 0) for s in _SCENARIOS}
