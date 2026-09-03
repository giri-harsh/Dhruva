"""Baselines B1-B5 (PRD §6.3). Every runnable baseline goes through the SAME
harness (`run_baselines.py`) with identical outage segments, identical ground
truth, identical metric definitions — that's the whole point of §6.3.

  B1  constant-velocity extrapolation        RUNNABLE — here
  B2  strapdown INS, no learning             Kamal's reference/anchor_ref (Python);
                                             we consume its per-outage trajectory
  B3  ESKF + NHC + ZUPT, no learned velocity Kamal's filter; our ablation runner
                                             plugs it in once it exists
  B4  WhONet                                 cited only (needs wheel-speed HW)
  B5  AVNet / DMDVDR                          cited only (the real bar: 0.64% drift)

Each baseline implements `predict_outage(seq, outage) -> (east, north, heading_end)`
in the local ENU frame whose origin is the outage's first row.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..contract import SAMPLE_RATE_HZ
from ..eval.geo import integrate_speed_heading, lla_to_local_enu

DT_S = 1.0 / SAMPLE_RATE_HZ


class Baseline(Protocol):
    id: str
    name: str
    runnable: bool

    def predict_outage(self, seq, outage) -> tuple[np.ndarray, np.ndarray, float]:
        ...


def truth_enu(seq, outage):
    """VBOX ground-truth path over the outage, ENU, origin at outage start row."""
    d = seq.df
    a, b = outage.start_row, outage.stop_row
    lat = d["veh_gt_lat_deg"].to_numpy()[a:b + 1]
    lon = d["veh_gt_lon_deg"].to_numpy()[a:b + 1]
    e, n = lla_to_local_enu(lat, lon, lat[0], lon[0])
    heading_end = np.radians(d["veh_heading_deg"].to_numpy()[b])
    return e, n, heading_end


class B1ConstantVelocity:
    id = "B1"
    name = "constant-velocity extrapolation (last GNSS velocity, held)"
    runnable = True

    def predict_outage(self, seq, outage):
        d = seq.df
        i0 = outage.start_row
        # "last GNSS velocity": speed + course at the sample before the outage
        v0 = float(d["veh_speed_mps"].to_numpy()[max(i0 - 1, 0)])
        hdg0 = float(np.radians(d["veh_heading_deg"].to_numpy()[max(i0 - 1, 0)]))
        n = outage.n_rows
        speed = np.full(n, v0)
        heading = np.full(n, hdg0)
        e, nth = integrate_speed_heading(speed, heading, DT_S, p0=(0.0, 0.0))
        return e, nth, hdg0


ALL_BASELINES: list[Baseline] = [B1ConstantVelocity()]
