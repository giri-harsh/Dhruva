"""B3 -- Error-State Kalman Filter + NHC + ZUPT (+ optional learned velocity
fusion), the Python reference `eskf_dead_reckon` named by this package's
README. Kamal's; conceptually ports the design already built and reviewed
in `core/src/main/kotlin/org/anchor/{fusion,orchestrator}` (propagate/
correct split, Joseph-form covariance, a generic gated measurement-update
path, chi-square innovation gating, NHC suppressed by a stationarity
detector which hands off to ZUPT) -- NOT a line-by-line translation, because
the input this filter sees is a genuinely different shape than the Kotlin
engine's: `feat_window` is Stage-2 (`contracts/frame_convention.md`),
vehicle-frame, GRAVITY-REMOVED linear acceleration, identical to what
ANCHOR-Net itself consumes. There is no gravity signal in this input to
mechanize or correct 3-D attitude from -- the data was already levelled and
rotated upstream by `features.py`'s own alignment step. A full 3-D 15-state
port is therefore not meaningful against this input; this filter's state is
`strapdown.py`'s own reduced (east, north, v_east, v_north, heading) model,
extended with exactly the states needed for a genuine ESKF around it:
accelerometer bias (forward, lateral) and gyroscope bias (yaw).

State (nominal == error state here -- heading is a scalar angle, not a
quaternion, so there is no 15-vs-16-style redundant-parameter gap):
    x = [p_e, p_n, v_e, v_n, heading, b_af, b_al, b_g]
      p_e, p_n   : position, world ENU, metres
      v_e, v_n   : velocity, world ENU, m/s  (NAV frame, matching the
                   Kotlin engine's own NominalState.velocity convention;
                   body-frame forward/lateral velocity is derived on
                   demand for NHC/velocity-fusion, exactly like
                   NominalState.bodyVelocity() there)
      heading    : world heading, radians, COMPASS convention (clockwise
                   from north) -- matches this package's own
                   `heading0_rad` contract and eval/geo.py's
                   integrate_speed_heading.
      b_af, b_al : accelerometer bias, body frame (forward, lateral), m/s^2
      b_g        : gyroscope bias (yaw rate), rad/s

## Two genuine cross-track findings from building this, both documented
## here rather than silently worked around:

1. **A real, previously-uncaught bug in `strapdown.py`'s (B2) body->world
   rotation**, found while deriving this file's own rotation and verified
   numerically (not by inspection): `a_e = ax*sin(h) + ay*cos(h-pi/2)`,
   `a_n = ax*cos(h) + ay*sin(h-pi/2)` is NOT a valid rotation -- feeding it
   a FIXED-magnitude vector (ax=1, ay=1, |a|=sqrt(2)) at headings 0/30/45/
   60/90/135/180 degrees returns |result| = 0.0/1.0/1.41/1.73/2.0/1.41/0.0
   respectively. A rotation must preserve vector length regardless of
   heading; this one does not. It happens to be invisible in
   `strapdown.py`'s own existing tests because both only ever exercise the
   forward axis alone (ay=0 throughout `test_baselines_b2.py`) -- the bug
   only manifests when forward and lateral acceleration are simultaneously
   nonzero, i.e. essentially every real turn. NOT fixed here: `strapdown.py`
   is Harshit's, frozen, and this is not a blocking incompatibility for
   `eskf_dead_reckon` (a separate function, its own rotation code) --
   flagged for Harshit/the team to decide on separately. This file uses the
   standard, magnitude-preserving 2-D rotation instead (see
   `_rotate_body_to_world` below), so B3 does not inherit B2's defect.

2. **No pre-outage calibration data reaches `eskf_dead_reckon`.**
   `strapdown_dead_reckon` resolves its gyro-to-heading sign empirically
   per-sequence from `gyro_z_pre`/`heading_rate_pre_radps` (see its own
   `_yaw_sign`); this function's frozen signature (this package's README)
   carries neither. `HEADING_RATE_SIGN` below is therefore a FIXED,
   geometrically-derived constant rather than a per-call calibration:
   ISO 8855's right-handed vehicle frame (x=forward, y=left, z=up) makes a
   positive gyro_z (right-hand rule about vehicle-up) a turn from forward
   toward left, i.e. counter-clockwise viewed from above, i.e. a
   DECREASING compass (clockwise-from-north) heading. Independently
   cross-checked against `ml/anchor/eval/anchornet_dr.py`'s own comment:
   "a right-handed +z (up) yaw is a LEFT turn => heading decreases." Both
   `align_sequence_to_vehicle_frame`'s levelling and yaw-search steps are
   proper (determinant +1, handedness-preserving) rotations, so this fixed
   sign should hold across sequences, not just the one anchornet_dr.py's
   comment was written against -- verified numerically in this file's own
   synthetic turn scenario (see `_selftest_turn` / the module's tests).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- layout --

DIM = 8
P_E, P_N, V_E, V_N, HEADING, B_AF, B_AL, B_G = range(DIM)

HEADING_RATE_SIGN = -1.0  # see module docstring, finding 2

# ------------------------------------------------------- process noise ----
# Same honest-default discipline as core/.../fusion/ProcessNoiseConfig.kt:
# genuinely unmeasured on real phone hardware, [VERIFY] against real data:
# these exist so propagation has *something* principled to run against, not
# a stand-in for calibration.
ACCEL_NOISE_DENSITY = 1.0e-3       # (m/s^2)/sqrt(Hz), isotropic fwd/lat
GYRO_NOISE_DENSITY = 1.0e-4        # (rad/s)/sqrt(Hz)
ACCEL_BIAS_RANDOM_WALK = 1.0e-5    # (m/s^2)/sqrt(Hz)
GYRO_BIAS_RANDOM_WALK = 1.0e-6     # (rad/s)/sqrt(Hz)

# Initial covariance at t=0 -- "GNSS reanchoring" in this interface's own
# terms: v0_mps/heading0_rad ARE the reanchor event (the last GNSS fix
# before the outage began), so what a real reanchor needs is a genuinely
# uncertain initial covariance around them, not a point estimate treated
# as exact. The interface provides no mid-outage GNSS channel to mask or
# reanchor against -- there is nothing else to respect; the outage is a
# total blackout by the ablation harness's own construction
# (ml/anchor/eval/outages.py), and this function reads only feat_window
# for its whole duration, exactly matching that.
INITIAL_POS_STD_M = 3.0
INITIAL_VEL_STD_MPS = 0.5
INITIAL_HEADING_STD_RAD = 0.05
INITIAL_ACCEL_BIAS_STD = 0.05
INITIAL_GYRO_BIAS_STD = 0.01

# ---------------------------------------------------- stationarity/ZUPT ---
# Conceptually StationarityDetector.kt: trace of the sample covariance
# (per-axis variance, not scalar-magnitude variance -- same blind-spot
# reasoning documented there) over a rolling window, thresholded.
STATIONARITY_WINDOW = 10           # samples, 1.0s at 10 Hz
STATIONARITY_ACCEL_VAR_THRESHOLD = 0.02   # (m/s^2)^2, [VERIFY]
STATIONARITY_GYRO_VAR_THRESHOLD = 0.001   # (rad/s)^2, [VERIFY]
# The model-displacement half of FR-26's own two-part trigger ("IMU energy
# below threshold AND near-zero predicted displacement" -- see
# StationarityDetector.kt's own doc, which explicitly flags this second
# half as a later extension once a live velocity stream is available to
# AND against). Here it is available (vel_mean_mps), so it is wired in:
# a *stationary-looking* IMU window is still treated as ZUPT-eligible only
# if the filter's own current forward-speed estimate is also this small --
# found necessary by running the velocity-fusion scenario, not by
# inspection: without it, ZUPT re-fires every tick during a genuinely
# IMU-quiet-but-actually-moving window (idling in traffic, say) and
# discards the previous tick's velocity-fusion correction before it can
# ever accumulate, making the model's confident reading pointless exactly
# in the stop-start regime it matters most for.
STATIONARY_SPEED_THRESHOLD_MPS = 0.5   # [VERIFY]

# ------------------------------------------------------------ NHC/ZUPT R --
NHC_LATERAL_NOISE_VARIANCE = 0.05   # (m/s)^2, matches NhcUpdate.kt's own default
ZUPT_VELOCITY_NOISE_VARIANCE = 0.01  # (m/s)^2, matches ZuptUpdate.kt's own default
VELOCITY_FIXED_R = 1.0              # (m/s)^2, used only when vel_logvar is absent

# FR-28's own principle ("a wrong update can never make the filter
# certain"), applied here to velocity: many consecutive ZUPT/NHC
# corrections during a long stop can otherwise drive velocity covariance
# so low that a genuinely correct but large velocity-fusion correction
# (the vehicle actually pulling away) reads as a chi-square outlier and is
# rejected -- found by running the velocity-fusion scenario against a
# speed change the un-floored filter had made itself too confident to
# accept, not by inspection. This floor is a lower bound only; it never
# INJECTS uncertainty on its own, it just refuses to let P shrink past it.
VELOCITY_VARIANCE_FLOOR = 1.0       # (m/s)^2, [VERIFY]

# ---------------------------------------------------------- chi-square ----
# Same sourced table as ChiSquareGate.kt -- standard chi-square critical
# values, not computed/approximated. dof=1 (NHC, VelocityUpdate), dof=2 (ZUPT).
_CHI2_P95 = {1: 3.841, 2: 5.991}


def _rotate_body_to_world(fwd, lat, heading):
    """Standard, magnitude-preserving 2-D rotation. See module docstring
    finding 1 for why this is NOT strapdown.py's own formula."""
    s, c = np.sin(heading), np.cos(heading)
    east = fwd * s - lat * c
    north = fwd * c + lat * s
    return east, north


def _rotate_world_to_body(east, north, heading):
    """Inverse of _rotate_body_to_world -- the transpose, since it's orthogonal."""
    s, c = np.sin(heading), np.cos(heading)
    fwd = east * s + north * c
    lat = -east * c + north * s
    return fwd, lat


@dataclass
class _Filter:
    x: np.ndarray   # [DIM]
    P: np.ndarray   # [DIM, DIM]

    def propagate(self, accel_fwd_meas: float, accel_lat_meas: float, gyro_meas: float, dt: float) -> None:
        h = self.x[HEADING]
        a_f = accel_fwd_meas - self.x[B_AF]
        a_l = accel_lat_meas - self.x[B_AL]
        g = gyro_meas - self.x[B_G]

        a_e, a_n = _rotate_body_to_world(a_f, a_l, h)

        self.x[P_E] += self.x[V_E] * dt + 0.5 * a_e * dt * dt
        self.x[P_N] += self.x[V_N] * dt + 0.5 * a_n * dt * dt
        self.x[V_E] += a_e * dt
        self.x[V_N] += a_n * dt
        self.x[HEADING] = h + HEADING_RATE_SIGN * g * dt
        # biases: nominal value unchanged (random walk -- Q carries their uncertainty)

        s, c = np.sin(h), np.cos(h)
        Fc = np.zeros((DIM, DIM))
        Fc[P_E, V_E] = 1.0
        Fc[P_N, V_N] = 1.0
        Fc[V_E, B_AF] = -s
        Fc[V_E, B_AL] = c
        Fc[V_E, HEADING] = a_n
        Fc[V_N, B_AF] = -c
        Fc[V_N, B_AL] = -s
        Fc[V_N, HEADING] = -a_e
        Fc[HEADING, B_G] = 1.0

        Phi = np.eye(DIM) + Fc * dt

        Qd = np.zeros((DIM, DIM))
        Qd[V_E, V_E] = Qd[V_N, V_N] = (ACCEL_NOISE_DENSITY ** 2) * dt
        Qd[HEADING, HEADING] = (GYRO_NOISE_DENSITY ** 2) * dt
        Qd[B_AF, B_AF] = Qd[B_AL, B_AL] = (ACCEL_BIAS_RANDOM_WALK ** 2) * dt
        Qd[B_G, B_G] = (GYRO_BIAS_RANDOM_WALK ** 2) * dt

        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)
        self._apply_velocity_floor()

    def _apply_velocity_floor(self) -> None:
        """See VELOCITY_VARIANCE_FLOOR's own comment. Floors only the
        velocity diagonal, in place, after any step that could shrink it."""
        self.P[V_E, V_E] = max(self.P[V_E, V_E], VELOCITY_VARIANCE_FLOOR)
        self.P[V_N, V_N] = max(self.P[V_N, V_N], VELOCITY_VARIANCE_FLOOR)

    def correct(self, H: np.ndarray, R: np.ndarray, innovation: np.ndarray) -> tuple[float, float]:
        """Joseph-form update (ErrorStateEkf.kt's own correct(), ported).
        Returns (mahalanobis_statistic, chi2_threshold) so the caller can
        gate BEFORE applying -- see `_gated_correct`."""
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        statistic = float(innovation @ S_inv @ innovation)
        dof = H.shape[0]
        threshold = _CHI2_P95.get(dof)
        if threshold is None:
            raise ValueError(f"no tabulated chi-square critical value for dof={dof}")
        return statistic, threshold, S_inv

    def apply_correction(self, H: np.ndarray, R: np.ndarray, innovation: np.ndarray, S_inv: np.ndarray) -> None:
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ innovation
        I_KH = np.eye(DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._apply_velocity_floor()


def _gated_correct(f: _Filter, H: np.ndarray, R: np.ndarray, innovation: np.ndarray) -> bool:
    """GatedMeasurementUpdate.kt, ported: compute the innovation statistic,
    gate it through the chi-square table, only apply if accepted. Returns
    whether the correction was applied."""
    statistic, threshold, S_inv = f.correct(H, R, innovation)
    accepted = statistic <= threshold
    if accepted:
        f.apply_correction(H, R, innovation, S_inv)
    return accepted


def _nhc_update(f: _Filter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FR-10: body-frame lateral velocity observed as zero."""
    h = f.x[HEADING]
    s, c = np.sin(h), np.cos(h)
    v_e, v_n = f.x[V_E], f.x[V_N]
    v_lat_pred = -v_e * c + v_n * s
    H = np.zeros((1, DIM))
    H[0, V_E] = -c
    H[0, V_N] = s
    H[0, HEADING] = v_e * s + v_n * c   # d(v_lat)/d(heading)
    R = np.array([[NHC_LATERAL_NOISE_VARIANCE]])
    innovation = np.array([0.0 - v_lat_pred])
    return H, R, innovation


def _zupt_update(f: _Filter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FR-26: world-frame velocity observed as zero directly -- no heading
    dependency, matching ZuptUpdate.kt's own simplest-possible Jacobian."""
    H = np.zeros((2, DIM))
    H[0, V_E] = 1.0
    H[1, V_N] = 1.0
    R = np.diag([ZUPT_VELOCITY_NOISE_VARIANCE, ZUPT_VELOCITY_NOISE_VARIANCE])
    innovation = np.array([0.0 - f.x[V_E], 0.0 - f.x[V_N]])
    return H, R, innovation


def _velocity_update(f: _Filter, mean_mps: float, variance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FR-11: ANCHOR-Net's forward-speed head, R = variance (already
    exp(log_variance) by the time it reaches here -- see eskf_dead_reckon).
    DO NOT treat mean_mps as ground truth: it is gated like everything else."""
    h = f.x[HEADING]
    s, c = np.sin(h), np.cos(h)
    v_e, v_n = f.x[V_E], f.x[V_N]
    v_fwd_pred = v_e * s + v_n * c
    H = np.zeros((1, DIM))
    H[0, V_E] = s
    H[0, V_N] = c
    H[0, HEADING] = v_e * c - v_n * s   # d(v_fwd)/d(heading)
    R = np.array([[variance]])
    innovation = np.array([mean_mps - v_fwd_pred])
    return H, R, innovation


def _broadcast_to_samples(values: np.ndarray, n_samples: int) -> np.ndarray:
    """vel_mean_mps/vel_logvar arrive "per inference window" (this
    package's README), not necessarily one entry per 10 Hz sample --
    ANCHOR-Net's own window is 2.0s/20 samples (contracts/model_io), and
    ml/anchor/eval/anchornet_dr.py's own predict_outage() holds exactly
    one Head A/B pair across each such window. This function is
    deliberately self-contained (no import from ml/anchor/contract) so
    reference/anchor_ref/ stays what its own README says it must: a clean,
    dependency-light signature a Kotlin port can match. If len(values)
    already equals n_samples, it is used as-is (already per-sample);
    otherwise each entry is held across an equal share of n_samples,
    the same "hold across the window" semantics anchornet_dr.py uses."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == n_samples:
        return values
    if len(values) == 0:
        return np.zeros(n_samples)
    window_idx = np.minimum(
        (np.arange(n_samples) / (n_samples / len(values))).astype(int),
        len(values) - 1,
    )
    return values[window_idx]


def _stationary_mask(feat_window: np.ndarray) -> np.ndarray:
    """Rolling-window per-axis variance over (accel_fwd, accel_lat, gyro_z)
    -- StationarityDetector.kt's own energy metric, ported. True where the
    window ending at that sample reads as stationary; False (including
    "not enough history yet") for the first STATIONARITY_WINDOW-1 samples,
    same conservative default as the Kotlin detector."""
    T = len(feat_window)
    w = STATIONARITY_WINDOW
    mask = np.zeros(T, dtype=bool)
    accel = feat_window[:, [0, 1]]
    gyro = feat_window[:, [5]]
    for t in range(w - 1, T):
        a_win = accel[t - w + 1:t + 1]
        g_win = gyro[t - w + 1:t + 1]
        a_energy = float(np.sum(np.var(a_win, axis=0)))
        g_energy = float(np.sum(np.var(g_win, axis=0)))
        mask[t] = a_energy < STATIONARITY_ACCEL_VAR_THRESHOLD and g_energy < STATIONARITY_GYRO_VAR_THRESHOLD
    return mask


def eskf_dead_reckon(
    feat_window: np.ndarray,
    *,
    dt_s: float,
    v0_mps: float,
    heading0_rad: float,
    vel_mean_mps: np.ndarray | None = None,
    vel_logvar: np.ndarray | None = None,
) -> dict:
    """B3: ESKF + NHC + ZUPT (+ optional gated velocity-head fusion).

    feat_window: [T, 6] aligned vehicle-frame accel(linear, m/s^2) + gyro
    (rad/s), IDENTICAL to what ANCHOR-Net receives. vel_mean_mps/vel_logvar,
    when given, are per-sample (already broadcast/held across their
    inference window by the caller -- this function does not know about
    inference-window boundaries) forward-speed mean and log-variance;
    R = exp(vel_logvar) per-sample, gated exactly like NHC/ZUPT, never
    trusted as ground truth.
    """
    f = feat_window.astype(np.float64)
    T = len(f)
    accel_fwd, accel_lat, gyro_z = f[:, 0], f[:, 1], f[:, 5]

    filt = _Filter(
        x=np.array([0.0, 0.0, v0_mps * np.sin(heading0_rad), v0_mps * np.cos(heading0_rad),
                    heading0_rad, 0.0, 0.0, 0.0]),
        P=np.diag([INITIAL_POS_STD_M ** 2, INITIAL_POS_STD_M ** 2,
                   INITIAL_VEL_STD_MPS ** 2, INITIAL_VEL_STD_MPS ** 2,
                   INITIAL_HEADING_STD_RAD ** 2,
                   INITIAL_ACCEL_BIAS_STD ** 2, INITIAL_ACCEL_BIAS_STD ** 2,
                   INITIAL_GYRO_BIAS_STD ** 2]),
    )

    stationary = _stationary_mask(f)

    east = np.empty(T + 1)
    north = np.empty(T + 1)
    east[0], north[0] = filt.x[P_E], filt.x[P_N]

    have_vel = vel_mean_mps is not None and vel_logvar is not None
    if have_vel:
        vel_mean_mps = _broadcast_to_samples(vel_mean_mps, T)
        vel_logvar = _broadcast_to_samples(vel_logvar, T)

    for t in range(T):
        filt.propagate(float(accel_fwd[t]), float(accel_lat[t]), float(gyro_z[t]), dt_s)

        # Velocity fusion runs BEFORE the ZUPT-vs-NHC dispatch decision (not
        # after, as a first draft of this loop had it) precisely so that
        # decision can see its effect -- see STATIONARY_SPEED_THRESHOLD_MPS's
        # own comment for why the ordering matters, found by running this
        # scenario, not by inspection.
        if have_vel:
            variance = float(np.exp(vel_logvar[t]))
            H, R, innovation = _velocity_update(filt, float(vel_mean_mps[t]), variance)
            _gated_correct(filt, H, R, innovation)

        v_fwd_now, _ = _rotate_world_to_body(filt.x[V_E], filt.x[V_N], filt.x[HEADING])
        is_stationary = bool(stationary[t]) and abs(v_fwd_now) < STATIONARY_SPEED_THRESHOLD_MPS

        if is_stationary:
            H, R, innovation = _zupt_update(filt)
            _gated_correct(filt, H, R, innovation)
            # FR-10: NHC suppressed while stationary -- ZUPT takes over.
        else:
            H, R, innovation = _nhc_update(filt)
            _gated_correct(filt, H, R, innovation)

        east[t + 1], north[t + 1] = filt.x[P_E], filt.x[P_N]

    return {"east_m": east, "north_m": north, "heading_end_rad": float(filt.x[HEADING])}
