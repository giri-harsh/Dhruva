"""B2 — strapdown INS dead-reckoning, no learning (PRD §6.3).

The physics-only path: given the vehicle-frame linear acceleration and gyro
(the SAME 6-channel aligned input ANCHOR-Net sees — so the comparison isolates
'integrate' vs 'perceive'), propagate heading by integrating yaw rate and
position by double-integrating forward/lateral acceleration, from an initial
pose taken at the last GNSS fix.

Error grows with the square of time — that is the whole point of the baseline
(PRD §1.2). No NHC, no ZUPT, no ground-truth reads during the outage.

Contract for a Kotlin port: state = (east, north, v_east, v_north, heading);
per 10 Hz step, in order:
    heading += sign * gyro_z * dt
    a_world  = Rz(heading) @ [accel_x, accel_y]        # x=fwd, y=left -> ENU
    v       += a_world * dt
    p       += v * dt
`sign` resolves the yaw-rate polarity from the pre-outage GNSS heading (last
known calibration), identical to anchor_ref has no CAN available.
"""
from __future__ import annotations

import numpy as np


def _yaw_sign(gyro_z_pre: np.ndarray, heading_rate_pre: np.ndarray) -> float:
    m = np.isfinite(gyro_z_pre) & np.isfinite(heading_rate_pre)
    if m.sum() < 20 or np.std(gyro_z_pre[m]) < 1e-4 or np.std(heading_rate_pre[m]) < 1e-4:
        return 1.0
    return 1.0 if np.corrcoef(gyro_z_pre[m], heading_rate_pre[m])[0, 1] >= 0 else -1.0


def strapdown_dead_reckon(
    feat_window: np.ndarray,          # [T, 6] aligned vehicle-frame: ax,ay,az,gx,gy,gz
    *,
    dt_s: float,
    v0_mps: float,                    # last GNSS speed
    heading0_rad: float,              # last GNSS heading (compass, CW from north)
    gyro_z_pre: np.ndarray | None = None,
    heading_rate_pre_radps: np.ndarray | None = None,
) -> dict:
    """Returns {'east_m', 'north_m', 'heading_end_rad'} over the outage,
    east/north length T+1 (includes the start point at the origin)."""
    f = np.asarray(feat_window, dtype=np.float64)
    T = len(f)
    ax, ay, gz = f[:, 0], f[:, 1], f[:, 5]

    sign = 1.0
    if gyro_z_pre is not None and heading_rate_pre_radps is not None:
        sign = _yaw_sign(np.asarray(gyro_z_pre, float),
                         np.asarray(heading_rate_pre_radps, float))

    heading = heading0_rad + sign * np.cumsum(gz) * dt_s        # [T]
    heading = np.concatenate([[heading0_rad], heading])[:T]     # heading at step start

    # compass heading (CW from north): east = sin(h), north = cos(h)
    ce, se = np.cos(heading), np.sin(heading)
    # rotate body [fwd, left] -> world [E, N]:  E = fwd*sin(h) - left*cos(h)? use
    # fwd along heading, left 90deg CCW of heading (= heading - 90 in compass)
    a_e = ax * se + ay * np.cos(heading - np.pi / 2)
    a_n = ax * ce + ay * np.sin(heading - np.pi / 2)

    v_e = v0_mps * np.sin(heading0_rad) + np.cumsum(a_e) * dt_s
    v_n = v0_mps * np.cos(heading0_rad) + np.cumsum(a_n) * dt_s
    east = np.concatenate([[0.0], np.cumsum(v_e) * dt_s])
    north = np.concatenate([[0.0], np.cumsum(v_n) * dt_s])
    return {"east_m": east, "north_m": north,
            "heading_end_rad": float(heading0_rad + sign * np.sum(gz) * dt_s)}
