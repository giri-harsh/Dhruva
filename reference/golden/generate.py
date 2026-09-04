"""Generate golden (input -> exact output) vectors for the reference engine
implementations in `reference/anchor_ref/`, so Kamal's Kotlin `core/` can
regression-test against fixed numbers (PRD §5.3, §10.5).

Same idea as `contracts/model_io/golden_vectors/` but for the FILTER/DR side.
Never hand-edit the output — change this script and re-run.

    python reference/golden/generate.py

Outputs (committed): reference/golden/strapdown_vectors.json
  and, once reference/anchor_ref/eskf.py exists, eskf_vectors.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from reference.anchor_ref import strapdown_dead_reckon  # noqa: E402

HERE = Path(__file__).parent
DT_S = 0.1


def _cases():
    """Fixed [T, 6] aligned vehicle-frame windows (accel_x/y/z, gyro_x/y/z)."""
    cases = {}

    T = 100
    z = np.zeros((T, 6))
    cases["straight_constant_10mps"] = dict(
        feat=z, dt_s=DT_S, v0_mps=10.0, heading0_rad=0.0)

    ramp = np.zeros((T, 6))
    ramp[:, 0] = 0.5                       # 0.5 m/s^2 forward
    cases["accelerating_from_rest"] = dict(
        feat=ramp, dt_s=DT_S, v0_mps=0.0, heading0_rad=0.0)

    turn = np.zeros((T, 6))
    turn[:, 5] = 0.1                       # constant 0.1 rad/s yaw
    cases["constant_yaw_cruise"] = dict(
        feat=turn, dt_s=DT_S, v0_mps=8.0, heading0_rad=1.0)

    rng = np.random.default_rng(11)
    cases["random_vibration"] = dict(
        feat=rng.normal(0, 0.4, (T, 6)), dt_s=DT_S, v0_mps=15.0, heading0_rad=2.5)

    return cases


def main() -> None:
    out = {"note": "Load reference/anchor_ref/strapdown_dead_reckon, run each "
                   "`input` through it, assert east_m/north_m/heading_end_rad "
                   "match `expected` within tolerance_abs. The Kotlin port of "
                   "the strapdown mechanization must reproduce these.",
           "tolerance_abs": 1e-6,
           "dt_s": DT_S,
           "vectors": []}
    for name, kw in _cases().items():
        r = strapdown_dead_reckon(kw["feat"], dt_s=kw["dt_s"], v0_mps=kw["v0_mps"],
                                  heading0_rad=kw["heading0_rad"])
        out["vectors"].append({
            "name": name,
            "input": {"feat_window": kw["feat"].round(6).tolist(),
                      "v0_mps": kw["v0_mps"], "heading0_rad": kw["heading0_rad"]},
            "expected": {"east_m": [round(v, 6) for v in r["east_m"]],
                         "north_m": [round(v, 6) for v in r["north_m"]],
                         "heading_end_rad": round(r["heading_end_rad"], 6)},
        })
    p = HERE / "strapdown_vectors.json"
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}  ({len(out['vectors'])} vectors)")


if __name__ == "__main__":
    main()
