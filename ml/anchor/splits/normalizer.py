"""Per-channel feature normaliser — fitted on the TRAIN split only.

PRD §6.2 hygiene rule 2 / §6.4: "normalisation statistics fitted on train only,
serialized with the model". The fitted mean/std become
`generate_stub_model.py`'s NORM_MEAN / NORM_STD constants and land in
`model_manifest.json`'s `normalization` block — the ONE place those numbers are
written down (contracts/model_io §2.2). Kamal's Kotlin applies exactly those.

`test_normaliser_fitted_on_train_only` in ml/tests/ asserts a normaliser fitted
on train produces stats that do NOT match a val/test refit — i.e. no leakage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contract import FEATURE_ORDER, NUM_FEATURES


@dataclass
class Normalizer:
    mean: np.ndarray          # shape [NUM_FEATURES]
    std: np.ndarray           # shape [NUM_FEATURES]
    feature_order: list[str]
    n_windows_fit: int
    fit_on: str               # provenance string, e.g. "split=train manifest=<sha>"

    @classmethod
    def fit(cls, windows_feat: np.ndarray, *, fit_on: str) -> "Normalizer":
        """windows_feat: [N, T, F] float array of RAW (pre-norm) model features
        from TRAIN windows only. Stats are over N*T samples per channel."""
        x = np.asarray(windows_feat, dtype=np.float64)
        if x.ndim != 3 or x.shape[2] != NUM_FEATURES:
            raise ValueError(f"expected [N, T, {NUM_FEATURES}], got {x.shape}")
        flat = x.reshape(-1, NUM_FEATURES)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-8] = 1.0            # guard a dead channel
        return cls(mean=mean, std=std, feature_order=list(FEATURE_ORDER),
                   n_windows_fit=int(x.shape[0]), fit_on=fit_on)

    def transform(self, windows_feat: np.ndarray) -> np.ndarray:
        x = np.asarray(windows_feat, dtype=np.float32)
        return (x - self.mean.astype(np.float32)) / self.std.astype(np.float32)

    def to_json(self) -> dict:
        return {
            "feature_order": self.feature_order,
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "n_windows_fit": self.n_windows_fit,
            "fit_on": self.fit_on,
            "formula": "(raw - mean) / std, per-feature, before inference",
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n",
                              encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["feature_order"] != list(FEATURE_ORDER):
            raise ValueError("normaliser feature_order does not match the contract")
        return cls(
            mean=np.array(d["mean"], dtype=np.float64),
            std=np.array(d["std"], dtype=np.float64),
            feature_order=d["feature_order"],
            n_windows_fit=d.get("n_windows_fit", -1),
            fit_on=d.get("fit_on", "unknown"),
        )
