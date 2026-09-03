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


import torch


@dataclass
class BatchAugmenter:
    """Vectorised augmentation applied to a whole [B, T, 6] tensor batch inside
    the training loop (far faster than per-window in a num_workers=0 DataLoader,
    which was ~40 s/epoch of the total). Operates on NORMALISED windows; the
    input is already vehicle-frame-aligned, so the rotation models residual
    ALIGNMENT error, not a full arbitrary mount — hence the smaller default angle.
    """
    # A controlled 4-way run (ml/docs/training-notes.md) showed heavy augmentation
    # slightly HURTS on this already-frame-aligned data (val RMSE 5.47 no-aug vs
    # 5.68 with the 7-deg version). Kept as a light regulariser only: small
    # residual-alignment rotation + additive noise, applied to a minority of
    # batches. Turn up (or off) via TrainConfig for the augmentation ablation row.
    rot_max_deg: float = 3.0
    noise_std_frac: float = 0.03
    gain_jitter_frac: float = 0.0
    bias_walk_frac: float = 0.0
    p_apply: float = 0.35

    def __call__(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        B, T, C = x.shape
        dev = x.device
        g = generator
        apply = torch.rand(B, generator=g, device=dev) < self.p_apply

        # small-angle SO(3) per sample: R ≈ I + [w]x for w ~ U(-a, a)^3
        a = torch.deg2rad(torch.tensor(self.rot_max_deg))
        w = (torch.rand(B, 3, generator=g, device=dev) * 2 - 1) * a
        zero = torch.zeros(B, device=dev)
        R = torch.stack([
            torch.stack([torch.ones(B, device=dev), -w[:, 2], w[:, 1]], -1),
            torch.stack([w[:, 2], torch.ones(B, device=dev), -w[:, 0]], -1),
            torch.stack([-w[:, 1], w[:, 0], torch.ones(B, device=dev)], -1),
        ], -2)  # [B,3,3]
        acc = torch.einsum("bij,btj->bti", R, x[:, :, 0:3])
        gyr = torch.einsum("bij,btj->bti", R, x[:, :, 3:6])
        xr = torch.cat([acc, gyr], -1)

        ch_std = x.std(dim=1, keepdim=True) + 1e-6
        noise = torch.randn(B, T, C, generator=g, device=dev) * self.noise_std_frac * ch_std
        gain = 1 + (torch.rand(B, 1, C, generator=g, device=dev) * 2 - 1) * self.gain_jitter_frac
        ramp = torch.linspace(0, 1, T, device=dev)[None, :, None]
        drift = ramp * (torch.rand(B, 1, C, generator=g, device=dev) * 2 - 1) * self.bias_walk_frac * ch_std
        xa = (xr + noise) * gain + drift

        return torch.where(apply[:, None, None], xa, x)


@dataclass
class Augmenter:
    """Per-window NumPy augmenter (kept for the reference/eval path). The
    training loop uses BatchAugmenter."""
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
