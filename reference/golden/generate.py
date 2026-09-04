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
import reference.anchor_ref as ref  # noqa: E402
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


def _eskf_cases():
    """Fixed [T, 6] windows exercising B3's own distinguishing behaviour --
    ZUPT (stationary), NHC (straight/turning), and gated velocity fusion --
    not a re-listing of _cases() above (B2 has no ZUPT/NHC/gate to exercise)."""
    cases = {}

    T = 100
    rng = np.random.default_rng(21)
    cases["stationary_zupt"] = dict(
        feat=rng.normal(0, 0.01, (T, 6)), dt_s=DT_S, v0_mps=0.0, heading0_rad=0.0)

    straight = np.zeros((T, 6))
    straight[:, 0] = 0.5 + 0.2 * np.sin(np.arange(T) * 1.3)
    cases["straight_driving_nhc"] = dict(
        feat=straight, dt_s=DT_S, v0_mps=0.0, heading0_rad=0.0)

    turn = np.zeros((T, 6))
    turn[:, 0] = 1.0
    turn[:, 5] = 0.2
    cases["sustained_turn"] = dict(
        feat=turn, dt_s=DT_S, v0_mps=0.0, heading0_rad=0.0)

    fused = np.zeros((T, 6))
    fused[:, 0] = 0.05  # tiny IMU-implied accel
    cases["gated_velocity_fusion"] = dict(
        feat=fused, dt_s=DT_S, v0_mps=3.0, heading0_rad=0.0,
        vel_mean_mps=np.full(T, 4.0), vel_logvar=np.full(T, np.log(0.05)))

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

    if not ref.HAS_ESKF:
        print("reference/anchor_ref/eskf.py not present -- skipping eskf_vectors.json")
        return

    eout = {"note": "Load reference/anchor_ref/eskf_dead_reckon, run each `input` "
                    "through it (vel_mean_mps/vel_logvar included when present), "
                    "assert east_m/north_m/heading_end_rad match `expected` within "
                    "tolerance_abs. The Kotlin port of the ESKF must reproduce these.",
            "tolerance_abs": 1e-6,
            "dt_s": DT_S,
            "vectors": []}
    for name, kw in _eskf_cases().items():
        call_kw = {"dt_s": kw["dt_s"], "v0_mps": kw["v0_mps"], "heading0_rad": kw["heading0_rad"]}
        if "vel_mean_mps" in kw:
            call_kw["vel_mean_mps"] = kw["vel_mean_mps"]
            call_kw["vel_logvar"] = kw["vel_logvar"]
        r = ref.eskf_dead_reckon(kw["feat"], **call_kw)
        vector_input = {"feat_window": kw["feat"].round(6).tolist(),
                        "v0_mps": kw["v0_mps"], "heading0_rad": kw["heading0_rad"]}
        if "vel_mean_mps" in kw:
            vector_input["vel_mean_mps"] = [round(v, 6) for v in kw["vel_mean_mps"]]
            vector_input["vel_logvar"] = [round(v, 6) for v in kw["vel_logvar"]]
        eout["vectors"].append({
            "name": name,
            "input": vector_input,
            "expected": {"east_m": [round(v, 6) for v in r["east_m"]],
                         "north_m": [round(v, 6) for v in r["north_m"]],
                         "heading_end_rad": round(r["heading_end_rad"], 6)},
        })
    ep = HERE / "eskf_vectors.json"
    ep.write_text(json.dumps(eout, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {ep}  ({len(eout['vectors'])} vectors)")


if __name__ == "__main__":
    main()
