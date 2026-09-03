"""Local ENU projection and trajectory helpers for evaluation.

Ground truth in IO-VNBD is VBOX lat/lon at 10 Hz. For outage-scale evaluation
(hundreds of metres, < 200 s) a local tangent-plane (equirectangular) projection
about the outage start point is accurate to well under the metre-class GT noise.
"""
from __future__ import annotations

import numpy as np

_EARTH_R = 6_371_000.0


def lla_to_local_enu(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """-> (east_m, north_m) arrays, origin at (lat0, lon0)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    east = (lon - lon0) * np.cos(lat0) * _EARTH_R
    north = (lat - lat0) * _EARTH_R
    return east, north


def integrate_speed_heading(speed_mps, heading_rad, dt_s, p0=(0.0, 0.0)):
    """Dead-reckon a path from a per-sample speed and heading (0 = +north,
    clockwise). Returns (east, north) arrays including the start point."""
    speed = np.asarray(speed_mps, dtype=np.float64)
    hdg = np.asarray(heading_rad, dtype=np.float64)
    de = speed * np.sin(hdg) * dt_s
    dn = speed * np.cos(hdg) * dt_s
    east = p0[0] + np.concatenate([[0.0], np.cumsum(de)])
    north = p0[1] + np.concatenate([[0.0], np.cumsum(dn)])
    return east, north


def path_length(east, north) -> float:
    return float(np.sum(np.hypot(np.diff(east), np.diff(north))))


def rigid_align_2d(src_e, src_n, dst_e, dst_n):
    """Least-squares rotation+translation (no scale) taking src onto dst
    (Umeyama). Returns aligned (e, n). Used for ATE."""
    src = np.c_[src_e, src_n].astype(np.float64)
    dst = np.c_[dst_e, dst_n].astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(S)
    D = np.diag([1.0, np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    t = mu_d - R @ mu_s
    out = (R @ src.T).T + t
    return out[:, 0], out[:, 1]
