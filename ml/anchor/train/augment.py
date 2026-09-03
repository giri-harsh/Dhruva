"""Window-level augmentation (PRD §6.4).

Implemented: static SO(3) rotation (arbitrary mounting), additive band-limited
noise, per-channel gain jitter (phone-model variation), simulated bias walk
(thermal drift), time-warp +/-5 %. NOT implemented on purpose: mirroring / time
reversal — a reversed drive teaches wrong dynamics.

Augmentation runs on the 6-channel [T, 6] vehicle-frame window (accel x/y/z,
gyro x/y/z) BEFORE normalisation. A rotation is applied jointly to the accel
triplet and the gyro triplet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rand_rotation(rng: np.random.Generator, max_angle_rad: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    angle = rng.uniform(-max_angle_rad, max_angle_rad)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass
class Augmenter:
    rot_max_deg: float = 12.0          # static mount perturbation
    noise_std_frac: float = 0.03       # additive noise, fraction of channel std
    gain_jitter_frac: float = 0.05     # per-channel multiplicative
    bias_walk_frac: float = 0.02       # slow drift across the window
    time_warp_frac: float = 0.05
    p_apply: float = 0.9

    def __call__(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p_apply:
            return window
        w = np.asarray(window, dtype=np.float64).copy()
        T = w.shape[0]

        # joint SO(3) on accel (0:3) and gyro (3:6)
        R = _rand_rotation(rng, np.radians(self.rot_max_deg))
        w[:, 0:3] = w[:, 0:3] @ R.T
        w[:, 3:6] = w[:, 3:6] @ R.T

        ch_std = w.std(axis=0) + 1e-6
        w += rng.normal(0.0, self.noise_std_frac, w.shape) * ch_std
        w *= (1.0 + rng.uniform(-self.gain_jitter_frac, self.gain_jitter_frac, w.shape[1]))
        drift = np.linspace(0, 1, T)[:, None] * rng.uniform(
            -self.bias_walk_frac, self.bias_walk_frac, w.shape[1]) * ch_std
        w += drift

        if self.time_warp_frac > 0:
            f = 1.0 + rng.uniform(-self.time_warp_frac, self.time_warp_frac)
            src = np.clip(np.linspace(0, (T - 1) * f, T), 0, T - 1)
            lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, T - 1)
            frac = (src - lo)[:, None]
            w = w[lo] * (1 - frac) + w[hi] * frac

        return w.astype(np.float32)
