"""ANCHOR-Net as a standalone dead-reckoner, for the outage bench.

This is NOT the shipped system (that is Kamal's ESKF consuming the velocity +
variance). It is the honest standalone measurement that goes next to B1 at the
Week-5 gate: "learned per-window forward speed + phone-IMU-integrated heading",
dead-reckoned through the outage.

  speed(t)   : Head A output for the window covering t (held across the window)
  heading(t) : last GNSS heading + integral of vehicle-frame yaw rate (gyro_z of
               the aligned phone IMU) — no GNSS, no CAN during the outage
  position   : integrate speed along heading at 10 Hz

FR-24 physical bounds (PRD §4.1 / §14.8): a window mean speed outside
[0, MAX_PLAUSIBLE_SPEED_MPS] is rejected and that window falls back to the
last accepted speed; the count is reported. These are the numbers Kamal's
ModelFallback.kt validates against — kept here and mirrored into
model_manifest.json when weights ship.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..contract import SAMPLE_RATE_HZ, WINDOW_SIZE_SAMPLES
from ..data.features import align_sequence_to_vehicle_frame, sequence_model_features
from ..models.anchornet import AnchorNet, AnchorNetConfig
from ..splits.normalizer import Normalizer
from .geo import integrate_speed_heading

DT_S = 1.0 / SAMPLE_RATE_HZ
MAX_PLAUSIBLE_SPEED_MPS = 100.0      # 360 km/h; > this in a 2 s window is impossible
MIN_PLAUSIBLE_SPEED_MPS = 0.0
GYRO_Z_FEATURE_INDEX = 5            # FEATURE_ORDER = [ax,ay,az,gx,gy,gz]; yaw = gz


@dataclass
class AnchorNetDeadReckoner:
    id: str = "ANCHORNET"
    name: str = "ANCHOR-Net: learned speed + IMU-integrated heading"
    runnable: bool = True

    checkpoint_path: str = ""
    normalizer_path: str = ""
    model_cfg: AnchorNetConfig | None = None
    heading_mode: str = "gyro"          # "gyro" (integrate aligned yaw rate) | "hold" (B1-style)
    speed_bias_mps: float = 0.0         # fixed additive speed correction, m/s
    online_bias: bool = True            # PRD §6.9: estimate the speed bias from the
                                        # pre-outage stretch where GNSS speed IS
                                        # available (last known good calibration —
                                        # not data from inside the outage) and
                                        # subtract it. The deployed residual monitor.
    bias_lookback_s: int = 45
    fuse: bool = True                   # 1-D constant-accel KF over consecutive windows,
                                        # gain from Head B's predicted variance — a
                                        # standalone stand-in for what Kamal's ESKF does

    def __post_init__(self):
        self._net = AnchorNet(self.model_cfg or AnchorNetConfig())
        self._net.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu", weights_only=True))
        self._net.eval()
        self._norm = Normalizer.load(self.normalizer_path)
        self._align_cache: dict[str, np.ndarray] = {}
        self.rejected_windows = 0

    def _features(self, seq) -> np.ndarray:
        if seq.seq_id not in self._align_cache:
            self._align_cache[seq.seq_id] = sequence_model_features(
                seq, align_sequence_to_vehicle_frame(seq))
        return self._align_cache[seq.seq_id]

    @torch.no_grad()
    def predict_outage(self, seq, outage):
        feats = self._features(seq)
        a, n = outage.start_row, outage.n_rows
        d = seq.df

        # per-window Head A + Head B, then optionally fuse consecutive windows
        w = WINDOW_SIZE_SAMPLES
        v0 = float(d["veh_speed_mps"].to_numpy()[max(a - 1, 0)])   # last GNSS speed
        bias = self.speed_bias_mps
        if self.online_bias:
            bias += self._estimate_speed_bias(seq, feats, a)
        # KF state [speed, accel], constant-accel model; measurement = window mean speed
        x_kf = np.array([v0, 0.0]); P = np.diag([1.0, 1.0])
        q_acc = 0.6 ** 2                              # process: accel wanders ~0.6 m/s^2 / window
        speed = np.empty(n, dtype=np.float64)
        last_ok = v0
        for si in range(0, n, w):
            win = feats[a + si:a + si + w]
            if len(win) < w:
                win = np.pad(win, ((0, w - len(win)), (0, 0)))
            xin = self._norm.transform(win[None]).astype(np.float32)
            out = self._net(torch.from_numpy(xin))
            z = float(out["velocity_mean_mps"]) - bias
            r = float(np.exp(out["velocity_log_variance"])) + 0.25    # Head B var + floor
            if not (MIN_PLAUSIBLE_SPEED_MPS <= z <= MAX_PLAUSIBLE_SPEED_MPS):
                self.rejected_windows += 1
                z = last_ok
            last_ok = z
            if self.fuse:
                dt = w * DT_S
                F = np.array([[1.0, dt], [0.0, 1.0]])
                x_kf = F @ x_kf
                P = F @ P @ F.T + np.array([[0.25 * dt ** 4, 0.5 * dt ** 3],
                                            [0.5 * dt ** 3, dt ** 2]]) * q_acc
                y = z - x_kf[0]; S = P[0, 0] + r
                K = P[:, 0] / S
                x_kf = x_kf + K * y
                P = P - np.outer(K, P[0, :])
                v_est = max(x_kf[0], 0.0)
                # fill the window with a linear speed profile [prev_end .. v_est]
                prev = speed[si - 1] if si > 0 else v0
                speed[si:si + w] = np.linspace(prev, v_est, min(w, n - si))
            else:
                speed[si:si + w] = max(z, 0.0)

        # heading: last GNSS heading + integral of aligned vehicle-frame yaw rate.
        # Compass heading is clockwise-from-north; a right-handed +z (up) yaw is a
        # LEFT turn => heading decreases. The sign of gyro_z out of features.py's
        # rotation is convention-dependent, so calibrate it once against the
        # PRE-outage stretch where GNSS heading is available (legitimate — it's
        # the last known good calibration, not data from inside the outage).
        hdg0 = float(np.radians(d["veh_heading_deg"].to_numpy()[max(a - 1, 0)]))
        if self.heading_mode == "hold":
            heading = np.full(n, hdg0)
        else:
            yaw_rate = feats[a:a + n, GYRO_Z_FEATURE_INDEX].astype(np.float64)
            sign = self._yaw_sign(seq, feats, a)
            heading = hdg0 + sign * np.cumsum(yaw_rate) * DT_S

        e, nth = integrate_speed_heading(speed, heading, DT_S, p0=(0.0, 0.0))
        return e, nth, float(heading[-1])

    def _estimate_speed_bias(self, seq, feats, outage_start: int) -> float:
        return self._estimate_speed_bias_from_arrays(
            feats, seq.df["veh_speed_mps"].to_numpy(), outage_start)

    @torch.no_grad()
    def _estimate_speed_bias_from_arrays(self, feats, gnss_speed, outage_start: int) -> float:
        """Median (model speed - GNSS speed) over inference-stride windows in the
        `bias_lookback_s` before the outage. The on-device residual monitor
        (PRD §6.9) — reads only rows strictly before `outage_start`."""
        w = WINDOW_SIZE_SAMPLES
        lo = max(0, outage_start - self.bias_lookback_s * SAMPLE_RATE_HZ)
        resid = []
        for si in range(lo, outage_start - w, w):
            win = feats[si:si + w]
            if len(win) < w:
                continue
            xin = self._norm.transform(win[None]).astype(np.float32)
            z = float(self._net(torch.from_numpy(xin))["velocity_mean_mps"])
            resid.append(z - float(np.mean(gnss_speed[si:si + w])))
        if len(resid) < 3:
            return 0.0
        return float(np.clip(np.median(resid), -8.0, 8.0))

    @staticmethod
    def _yaw_sign(seq, feats, outage_start: int, lookback_s: int = 60) -> float:
        d = seq.df
        lo = max(0, outage_start - lookback_s * SAMPLE_RATE_HZ)
        hi = outage_start
        if hi - lo < 5 * SAMPLE_RATE_HZ:
            return 1.0
        hdg = np.unwrap(np.radians(d["veh_heading_deg"].to_numpy()[lo:hi]))
        dh = np.diff(hdg)
        gz = feats[lo:hi - 1, GYRO_Z_FEATURE_INDEX].astype(np.float64)
        if np.std(gz) < 1e-4 or np.std(dh) < 1e-4:
            return 1.0
        # want sign * gz ~ dh (compass heading rate)
        return 1.0 if np.corrcoef(gz, dh)[0, 1] >= 0 else -1.0
