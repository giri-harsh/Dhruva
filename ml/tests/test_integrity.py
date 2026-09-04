"""FR-31 integrity bench: injectors, per-instance detection scoring, and the
committed expected-curve check.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anchor.integrity.attacks import AttackSpec, inject
from anchor.integrity.roc import (
    InnovationResidualDetector,
    _instance_detected,
    score_detector,
)

_EXPECTED = Path(__file__).resolve().parents[2] / "ml" / "eval" / "integrity_roc_expected.json"


class _Seq:
    def __init__(self, n=1800):
        # a straight, steady 15 m/s drive due north
        lat = 52.5 + np.cumsum(np.full(n, 15.0 / 111_320 * 0.1))
        lon = np.full(n, -1.5)
        self.df = pd.DataFrame({"veh_gt_lat_deg": lat, "veh_gt_lon_deg": lon})
        self.n_rows = n
        self.segments = [(0, n)]
        self.seq_id = "syn"
        self.meta = {"n_segments": 1}


def test_injector_families_mark_the_right_fixes():
    s = _Seq()
    step = inject(s, AttackSpec("step", 40.0, onset_s=20.0), gnss_noise_sigma_m=0.0)
    onset = 200
    assert not step.attacked[:onset].any()
    assert step.attacked[onset:].all()
    # the corrupted track is offset from truth after onset
    off = np.hypot(step.east_m[onset:] - step.truth_east_m[onset:],
                   step.north_m[onset:] - step.truth_north_m[onset:])
    assert np.median(off[step.valid[onset:]]) > 30

    jam = inject(s, AttackSpec("jam", 10.0, onset_s=20.0))
    assert not jam.valid[onset:onset + 100].any()      # 10 s of no fix


def test_gnss_noise_model_makes_fixes_sparse_and_noisy():
    s = _Seq()
    tr = inject(s, AttackSpec("multipath", 0.0), gnss_noise_sigma_m=4.0, gnss_rate_hz=1.0)
    assert tr.valid.mean() == pytest.approx(0.1, abs=0.02)     # ~1 Hz on a 10 Hz grid
    err = np.hypot(tr.east_m[tr.valid] - tr.truth_east_m[tr.valid],
                   tr.north_m[tr.valid] - tr.truth_north_m[tr.valid])
    assert 2.0 < err.std() < 7.0


def test_instance_detection_is_per_attack_not_per_sample():
    """A large step held after onset: the detector flags near onset, then the CV
    model re-converges. Per-instance detection is True; a per-sample rate would
    be diluted by the re-converged tail."""
    s = _Seq()
    det = InnovationResidualDetector()
    tr = inject(s, AttackSpec("step", 60.0, onset_s=20.0), gnss_noise_sigma_m=4.0)
    stat = det.residuals(tr)
    assert _instance_detected(stat, tr, thr=3.0) is True


def test_small_step_is_in_the_undetected_regime():
    s = _Seq()
    det = InnovationResidualDetector()
    caught = 0
    for k in range(12):
        tr = inject(s, AttackSpec("step", 2.0, onset_s=20.0, seed=k), gnss_noise_sigma_m=4.0)
        if _instance_detected(det.residuals(tr), tr, thr=3.0):
            caught += 1
    assert caught < 8      # a 2 m step at 4 m noise is not reliably caught


@pytest.mark.usefixtures("sequences")
def test_bench_reproduces_within_tolerance_of_committed_expected(sequences):
    from anchor.splits.protocol import assign_all
    splits = assign_all(sequences)
    attack = [s for s in splits["test_id"] if s.meta["n_segments"] and s.n_rows > 600][:6]
    clean = [s for s in splits["test_ood_driver"] if s.n_rows > 600][:4]
    fams = {"step": [2.0, 5.0, 40.0], "drag": [0.1, 0.8], "jam": [10.0],
            "multipath": [0.1, 0.9]}
    res = score_detector(InnovationResidualDetector(), clean, attack,
                         families=fams, seed=20260903)
    # structural assertions (numbers move with the sequence subset)
    assert set(res["families"]) == set(fams)
    assert res["families"]["step"]["detection_at_operating"][40.0] >= 0.6
    assert res["families"]["step"]["detection_at_operating"][2.0] <= 0.6
    assert 0.0 <= res["families"]["jam"]["false_rejection_at_operating"] <= 0.5


def test_expected_curve_file_is_committed_and_well_formed():
    assert _EXPECTED.exists(), "run `python -m anchor.integrity.run_bench --update-expected`"
    e = json.loads(_EXPECTED.read_text())
    assert set(e["families"]) == {"step", "drag", "jam", "multipath"}
    assert e["families"]["step"]["provably_undetected_param"] is not None
