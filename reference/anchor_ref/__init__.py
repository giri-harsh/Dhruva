"""`reference/anchor_ref/` — the Python reference for the on-device engine.

Two consumers (PRD §5.3, §10.5):
  1. The ML track's baseline/ablation harness (`ml/anchor/bench/`) runs B2/B3
     through here so every baseline uses identical inputs, ground truth, and
     metric definitions (§6.3).
  2. Kamal's Kotlin `core/` regression-tests against golden vectors this module
     generates (`reference/golden/`).

Harshit maintains this file; Kamal's Kotlin must reproduce its arithmetic. The
B2 strapdown mechanization below is complete and runnable now. `eskf` (B3) is
Kamal's — this package exposes the interface it must satisfy and marks B3
unavailable until `anchor_ref/eskf.py` lands.
"""
from __future__ import annotations

from .strapdown import strapdown_dead_reckon

try:  # B3 — Kamal's, optional until it exists
    from .eskf import eskf_dead_reckon  # noqa: F401
    HAS_ESKF = True
except Exception:  # pragma: no cover
    HAS_ESKF = False

__all__ = ["strapdown_dead_reckon", "HAS_ESKF"]
