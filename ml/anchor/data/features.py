"""Phone device-frame IMU -> vehicle-frame 6-channel model input.

This is the training-time equivalent of Kamal's on-device AlignmentService
(v3 PRD FR-04/FR-05). The exported model's contract input is exactly
`accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z` (contracts/model_io §2.2),
vehicle-frame, in the ISO-8855-style frame proposed in
`contracts/frame_convention.md`:  x = forward, y = left, z = up, right-handed.
`accel_*` here is LINEAR acceleration (gravity removed) — the phone ships a
`GRAVITY` channel, so `linear = accel - gravity`, and vehicle-frame `accel_z`
is then suspension/road motion, not a ~9.8 offset.

--- v0 alignment (per-sequence static mount) ---
1. roll & pitch: the mean phone GRAVITY vector points "down"; rotate so it
   lands on -z. Fixes 2 of 3 DOF, needs no vehicle data.
2. yaw: after step 1 the horizontal linear-accel vector still has an unknown
   in-plane rotation. Pick the yaw angle whose forward axis best matches the
   sign and shape of the vehicle's longitudinal acceleration
   (d/dt vehicle speed) over the sequence — a 1-D search maximising correlation.
3. gyro: rotate the 3-axis device-frame gyro by the same rotation.

Limitations (documented, not hidden): assumes a rigid mount for the whole
sequence; a loose phone breaks assumption 1. Sequences where step 2's best
correlation is weak are exactly the `quality.usability != "use"` ones. Per-window
mount variation and remounts are covered by SO(3) augmentation at train time
(PRD §6.4), not here. This will be refined once Kamal's AlignmentService
convention is locked (open decision #1).
"""
from __future__ import annotations

import numpy as np

from ..contract import FEATURE_ORDER

_G = 9.80665


def _rotation_gravity_to_down(gravity_xyz: np.ndarray) -> np.ndarray:
    """3x3 rotation R such that R @ mean_gravity is parallel to -z (0,0,-1)."""
    g = np.asarray(gravity_xyz, dtype=np.float64)
    g = g[np.all(np.isfinite(g), axis=1)]
    if len(g) == 0:
        return np.eye(3)
    gm = g.mean(axis=0)
    n = np.linalg.norm(gm)
    if n < 1e-6:
        return np.eye(3)
    src = gm / n
    dst = np.array([0.0, 0.0, -1.0])
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _yaw_rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _best_yaw(accel_lvl: np.ndarray, veh_long_accel: np.ndarray) -> float:
    """1-D search: yaw angle whose +x axis best matches vehicle longitudinal accel."""
    a = accel_lvl[:, :2]  # horizontal plane after levelling
    y = np.asarray(veh_long_accel, dtype=np.float64)
    m = np.all(np.isfinite(a), axis=1) & np.isfinite(y)
    if m.sum() < 200 or np.std(y[m]) < 1e-3:
        return 0.0
    a, y = a[m], y[m] - y[m].mean()
    best_theta, best_score = 0.0, -np.inf
    for theta in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        fwd = a @ np.array([np.cos(theta), np.sin(theta)])
        fwd = fwd - fwd.mean()
        denom = np.std(fwd) * np.std(y)
        score = float(np.mean(fwd * y) / denom) if denom > 0 else 0.0
        if score > best_score:
            best_score, best_theta = score, theta
    return best_theta


def align_sequence_to_vehicle_frame(seq) -> dict:
    """Returns {'R': 3x3, 'yaw_offset': float, 'yaw_fit_corr': float}."""
    d = seq.df
    grav = np.c_[d["phone_gravity_x_mps2"], d["phone_gravity_y_mps2"], d["phone_gravity_z_mps2"]]
    accel = np.c_[d["phone_accel_x_mps2"], d["phone_accel_y_mps2"], d["phone_accel_z_mps2"]]
    lin = accel - grav                      # device-frame linear accel

    R_level = _rotation_gravity_to_down(grav)
    lin_lvl = lin @ R_level.T

    veh_long = d["veh_long_accel_mps2"].to_numpy()
    theta = _best_yaw(lin_lvl, veh_long)
    R = _yaw_rotation(theta) @ R_level

    # report the fit quality
    fwd = (lin @ R.T)[:, 0]
    m = np.isfinite(fwd) & np.isfinite(veh_long)
    corr = (float(np.corrcoef(fwd[m], veh_long[m])[0, 1])
            if m.sum() > 200 and np.std(veh_long[m]) > 1e-3 else float("nan"))
    return {"R": R, "yaw_offset": float(theta), "yaw_fit_corr": corr}


def sequence_model_features(seq, alignment: dict | None = None) -> np.ndarray:
    """[T, 6] float32 vehicle-frame model input for the whole sequence, in
    contract FEATURE_ORDER (accel_x/y/z linear m/s^2, gyro_x/y/z rad/s)."""
    if alignment is None:
        alignment = align_sequence_to_vehicle_frame(seq)
    R = alignment["R"]
    d = seq.df
    grav = np.c_[d["phone_gravity_x_mps2"], d["phone_gravity_y_mps2"], d["phone_gravity_z_mps2"]]
    accel = np.c_[d["phone_accel_x_mps2"], d["phone_accel_y_mps2"], d["phone_accel_z_mps2"]]
    gyro = np.c_[d["phone_gyro_roll_radps"], d["phone_gyro_pitch_radps"], d["phone_gyro_yaw_radps"]]

    lin_v = (accel - grav) @ R.T
    gyro_v = gyro @ R.T
    feat = np.concatenate([lin_v, gyro_v], axis=1).astype(np.float32)
    assert feat.shape[1] == len(FEATURE_ORDER) == 6
    return feat
