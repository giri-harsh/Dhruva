"""Windowed training dataset for ANCHOR-Net.

Per sequence, once: run the frame alignment (features.py) to get the [T, 6]
vehicle-frame model input, and build the per-window displacement labels
(labels.py). Then the SequenceWindower defines which [start, stop) slices are
legal windows. Normalisation (train-fitted) is applied at __getitem__.

Aligned features are cached to `ml/.cache/features/<sha>.npy` keyed by the
source-CSV SHA-256 so repeated runs / seeds don't re-run the yaw search.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contract import SAMPLE_RATE_HZ, WINDOW_SIZE_SAMPLES
from ..data.features import align_sequence_to_vehicle_frame, sequence_model_features
from ..data.labels import SequenceLabeller
from ..splits.normalizer import Normalizer
from ..splits.windower import SequenceWindower
from .augment import Augmenter

_CACHE = Path(__file__).resolve().parents[3] / "ml" / ".cache" / "features"


def _cache_key(seq) -> str:
    m = seq.meta
    h = hashlib.sha256(f"{m['s_path']}|{m['v_path']}|v1".encode()).hexdigest()[:16]
    return h


def aligned_features(seq) -> np.ndarray:
    _CACHE.mkdir(parents=True, exist_ok=True)
    p = _CACHE / f"{seq.seq_id}_{_cache_key(seq)}.npy"
    if p.exists():
        return np.load(p)
    feats = sequence_model_features(seq, align_sequence_to_vehicle_frame(seq))
    np.save(p, feats)
    return feats


class AnchorWindowDataset(Dataset):
    def __init__(
        self,
        sequences,
        *,
        radius_m: float,
        normalizer: Normalizer,
        training: bool,
        augmenter: Augmenter | None = None,
        seed: int = 0,
    ):
        self.norm = normalizer
        self.training = training
        self.aug = augmenter if training else None
        self._rng = np.random.default_rng(seed)

        win = SequenceWindower(training=training)
        self.feats: dict[str, np.ndarray] = {}
        self.index: list[tuple[str, int, int]] = []
        self._target = []          # mean speed per window (precomputed once)
        self._sigma = []           # label sigma on SPEED per window (precomputed once)
        win_dur_s = WINDOW_SIZE_SAMPLES / SAMPLE_RATE_HZ
        for seq in sequences:
            if seq.meta.get("usability") == "drop":
                continue
            self.feats[seq.seq_id] = aligned_features(seq)
            lab = SequenceLabeller(seq, radius_m)
            for w in win.windows(seq):
                wl = lab.label(w.start, w.stop)
                self.index.append((seq.seq_id, w.start, w.stop))
                self._target.append(wl.mean_speed_mps)
                self._sigma.append(wl.label_sigma_m / win_dur_s)
        self._target = np.asarray(self._target, dtype=np.float32)
        self._sigma = np.asarray(self._sigma, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        seq_id, a, b = self.index[i]
        window = self.feats[seq_id][a:b].astype(np.float32)   # [20, 6]
        if window.shape[0] != WINDOW_SIZE_SAMPLES:
            window = np.pad(window, ((0, WINDOW_SIZE_SAMPLES - window.shape[0]), (0, 0)))
        if self.aug is not None:
            window = self.aug(window, self._rng)

        x = self.norm.transform(window[None])[0]             # [20, 6] normalised
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            "target_speed": torch.tensor(self._target[i], dtype=torch.float32),
            "label_sigma": torch.tensor(self._sigma[i], dtype=torch.float32),
            "seq_id": seq_id,
        }

    # speed-decile re-weighting sampler weights (PRD §6.6 class balance)
    def sample_weights(self) -> np.ndarray:
        speeds = self._target
        deciles = np.clip((speeds / (speeds.max() + 1e-6) * 10).astype(int), 0, 9)
        counts = np.bincount(deciles, minlength=10).astype(float)
        counts[counts == 0] = 1.0
        w = 1.0 / counts[deciles]
        return w / w.mean()
