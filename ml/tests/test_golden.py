import json
from pathlib import Path

import pytest

_MANIFEST = Path(__file__).resolve().parents[2] / "ml" / "golden" / "manifest.json"


def test_manifest_exists_and_is_well_formed():
    assert _MANIFEST.exists(), "run `python -m anchor.golden.build_golden`"
    m = json.loads(_MANIFEST.read_text())
    assert m["n_segments"] == len(m["segments"]) == 40
    assert sum(1 for s in m["segments"] if s["public_subset"]) == 10
    keys = [s["key"] for s in m["segments"]]
    assert len(set(keys)) == 40, "duplicate golden segments"
    # content hash matches
    import hashlib
    body = json.dumps(m["segments"], sort_keys=True).encode()
    assert hashlib.sha256(body).hexdigest() == m["content_sha256"]


def test_golden_segments_are_from_test_splits_only(sequences):
    from anchor.splits.protocol import assign_all
    splits = assign_all(sequences)
    allowed = {s.seq_id for name in ("test_id", "test_ood_driver", "test_repeat_corridor")
               for s in splits[name]}
    forbidden = {s.seq_id for name in ("train", "val") for s in splits[name]}
    m = json.loads(_MANIFEST.read_text())
    for s in m["segments"]:
        assert s["seq_id"] in allowed
        assert s["seq_id"] not in forbidden


def test_scenario_classifier_labels_are_in_the_known_set(sequences):
    from anchor.eval.scenarios import _SCENARIOS, classify_window
    s = next(x for x in sequences if x.n_rows > 5000)
    for start in range(500, s.n_rows - 600, 800):
        assert classify_window(s, start, start + 600) in _SCENARIOS


def test_scenario_hard_braking_detected():
    """Synthetic: a window with a −4 m/s^2 longitudinal spike is hard_braking."""
    import numpy as np
    import pandas as pd

    class S:
        pass
    n = 300
    df = pd.DataFrame({
        "veh_speed_mps": np.full(n, 15.0),
        "veh_yaw_rate_radps": np.zeros(n),
        "veh_long_accel_mps2": np.concatenate([np.zeros(150), [-4.0], np.zeros(149)]),
    })
    s = S(); s.df = df
    from anchor.eval.scenarios import classify_window
    assert classify_window(s, 0, n) == "hard_braking"
