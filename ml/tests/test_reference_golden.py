"""The committed reference/golden/strapdown_vectors.json must reproduce from the
current reference/anchor_ref/strapdown implementation (the same stale-artefact
guard contracts-ci applies to the model stub). Kamal's Kotlin port checks
against the same file.
"""
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def test_strapdown_golden_vectors_reproduce():
    from reference.anchor_ref import strapdown_dead_reckon

    doc = json.loads((_ROOT / "reference" / "golden" / "strapdown_vectors.json").read_text())
    tol = doc["tolerance_abs"]
    assert doc["vectors"], "no golden vectors"
    for v in doc["vectors"]:
        feat = np.array(v["input"]["feat_window"], dtype=np.float64)
        r = strapdown_dead_reckon(feat, dt_s=doc["dt_s"],
                                  v0_mps=v["input"]["v0_mps"],
                                  heading0_rad=v["input"]["heading0_rad"])
        ex = v["expected"]
        assert np.allclose(r["east_m"], ex["east_m"], atol=tol), f"{v['name']}: east drift"
        assert np.allclose(r["north_m"], ex["north_m"], atol=tol), f"{v['name']}: north drift"
        assert abs(r["heading_end_rad"] - ex["heading_end_rad"]) < tol


def test_eskf_golden_vectors_reproduce():
    from reference.anchor_ref import eskf_dead_reckon

    doc = json.loads((_ROOT / "reference" / "golden" / "eskf_vectors.json").read_text())
    tol = doc["tolerance_abs"]
    assert doc["vectors"], "no golden vectors"
    for v in doc["vectors"]:
        inp = v["input"]
        feat = np.array(inp["feat_window"], dtype=np.float64)
        kw = {"dt_s": doc["dt_s"], "v0_mps": inp["v0_mps"], "heading0_rad": inp["heading0_rad"]}
        if "vel_mean_mps" in inp:
            kw["vel_mean_mps"] = np.array(inp["vel_mean_mps"], dtype=np.float64)
            kw["vel_logvar"] = np.array(inp["vel_logvar"], dtype=np.float64)
        r = eskf_dead_reckon(feat, **kw)
        ex = v["expected"]
        assert np.allclose(r["east_m"], ex["east_m"], atol=tol), f"{v['name']}: east drift"
        assert np.allclose(r["north_m"], ex["north_m"], atol=tol), f"{v['name']}: north drift"
        assert abs(r["heading_end_rad"] - ex["heading_end_rad"]) < tol
