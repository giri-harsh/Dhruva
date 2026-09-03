"""Single source of truth for the model I/O contract, re-exported for the ML pipeline.

Per PRD-ML-BACKEND.md §7.2 ("Model I/O drift"): the training/eval/windowing code
MUST NOT redefine window size, feature order, sample rate, or output names locally.
It imports them from the contract generator script so a change there is a change
everywhere, and `contracts/model_io/test_contract.py` + `contracts-ci.yml` catch drift.

If you need to change any value here, you change it in
`contracts/model_io/generate_stub_model.py`, re-run that script, and follow
`contracts/VERSIONING.md` (message Kamal first for a MAJOR bump).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN = _REPO_ROOT / "contracts" / "model_io" / "generate_stub_model.py"

if not _GEN.exists():  # pragma: no cover - defensive
    raise RuntimeError(f"contract generator not found at {_GEN}")

_spec = importlib.util.spec_from_file_location("_anchor_contract_gen", _GEN)
_mod = importlib.util.module_from_spec(_spec)
# generate_stub_model.py imports numpy/onnx at module load; those are in requirements.txt
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

CONTRACT_VERSION: str = _mod.CONTRACT_VERSION
WINDOW_SIZE_SAMPLES: int = _mod.WINDOW_SIZE_SAMPLES
SAMPLE_RATE_HZ: int = _mod.SAMPLE_RATE_HZ
FEATURE_ORDER: list[str] = list(_mod.FEATURE_ORDER)
NUM_FEATURES: int = _mod.NUM_FEATURES
INPUT_NAME: str = _mod.INPUT_NAME
INPUT_SHAPE: list[int] = list(_mod.INPUT_SHAPE)
OUTPUT_MEAN_NAME: str = _mod.OUTPUT_MEAN_NAME
OUTPUT_LOGVAR_NAME: str = _mod.OUTPUT_LOGVAR_NAME

WINDOW_DURATION_S: float = WINDOW_SIZE_SAMPLES / SAMPLE_RATE_HZ  # 2.0 s

__all__ = [
    "CONTRACT_VERSION",
    "WINDOW_SIZE_SAMPLES",
    "SAMPLE_RATE_HZ",
    "FEATURE_ORDER",
    "NUM_FEATURES",
    "INPUT_NAME",
    "INPUT_SHAPE",
    "OUTPUT_MEAN_NAME",
    "OUTPUT_LOGVAR_NAME",
    "WINDOW_DURATION_S",
]
