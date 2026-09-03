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

        # per-10Hz-sample speed from windowed Head A
        speed = np.empty(n, dtype=np.float64)
        last_ok = float(d["veh_speed_mps"].to_numpy()[max(a - 1, 0)])
        w = WINDOW_SIZE_SAMPLES
        for s in range(0, n, w):
            win_lo = a + s
            win = feats[win_lo:win_lo + w]
            if len(win) < w:
                win = np.pad(win, ((0, w - len(win)), (0, 0)))
            x = self._norm.transform(win[None]).astype(np.float32)
            mean_speed = float(self._net(torch.from_numpy(x))["velocity_mean_mps"])
            if not (MIN_PLAUSIBLE_SPEED_MPS <= mean_speed <= MAX_PLAUSIBLE_SPEED_MPS):
                self.rejected_windows += 1
                mean_speed = last_ok
            last_ok = mean_speed
            speed[s:s + w] = mean_speed

        # heading: last GNSS heading + integral of aligned vehicle-frame yaw rate.
        # Compass heading is clockwise-from-north; a right-handed +z (up) yaw is a
        # LEFT turn => heading decreases. The sign of gyro_z out of features.py's
        # rotation is convention-dependent, so calibrate it once against the
        # PRE-outage stretch where GNSS heading is available (legitimate — it's
        # the last known good calibration, not data from inside the outage).
        hdg0 = float(np.radians(d["veh_heading_deg"].to_numpy()[max(a - 1, 0)]))
        yaw_rate = feats[a:a + n, GYRO_Z_FEATURE_INDEX].astype(np.float64)
        sign = self._yaw_sign(seq, feats, a)
        heading = hdg0 + sign * np.cumsum(yaw_rate) * DT_S

        e, nth = integrate_speed_heading(speed, heading, DT_S, p0=(0.0, 0.0))
        return e, nth, float(heading[-1])

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
