import numpy as np
import pytest

from anchor.bench.baselines import B1ConstantVelocity, truth_enu
from anchor.eval.geo import integrate_speed_heading
from anchor.eval.metrics import score_outage
from anchor.eval.outages import DURATIONS_S, sample_outages
from anchor.splits.protocol import assign_all


def test_integrate_straight_line():
    e, n = integrate_speed_heading(np.full(100, 10.0), np.zeros(100), 0.1)
    assert n[-1] == pytest.approx(100.0)   # 10 m/s * 10 s due north
    assert abs(e[-1]) < 1e-9


def test_score_outage_perfect_prediction_is_zero_drift():
    t = np.linspace(0, 300, 301)
    e = t.copy(); n = np.zeros_like(t)
    s = score_outage(seq_id="x", duration_s=30, scenario="s",
                     pred_e=e, pred_n=n, truth_e=e, truth_n=n)
    assert s.final_error_m == pytest.approx(0.0, abs=1e-6)
    assert s.drift_pct == pytest.approx(0.0, abs=1e-6)


def test_outage_sampler_deterministic(sequences):
    seqs = assign_all(sequences)["test_id"]
    a = sample_outages(seqs, seed=1)
    b = sample_outages(seqs, seed=1)
    c = sample_outages(seqs, seed=2)
    assert [o.key() for o in a] == [o.key() for o in b]
    assert [o.key() for o in a] != [o.key() for o in c]
    assert {o.duration_s for o in a} <= set(DURATIONS_S)


def test_b1_runs_and_drifts_more_over_longer_outages(sequences):
    seqs = assign_all(sequences)["test_id"]
    outs = sample_outages(seqs, seed=3)
    by_id = {s.seq_id: s for s in seqs}
    b1 = B1ConstantVelocity()
    drift_by_dur = {d: [] for d in DURATIONS_S}
    for o in outs:
        seq = by_id[o.seq_id]
        te, tn, thdg = truth_enu(seq, o)
        pe, pn, phdg = b1.predict_outage(seq, o)
        m = min(len(pe), len(te))
        sc = score_outage(seq_id=o.seq_id, duration_s=o.duration_s, scenario=o.scenario,
                          pred_e=pe[:m], pred_n=pn[:m], truth_e=te[:m], truth_n=tn[:m])
        if np.isfinite(sc.drift_pct):
            drift_by_dur[o.duration_s].append(sc.drift_pct)
    meds = {d: float(np.median(v)) for d, v in drift_by_dur.items() if v}
    # B1 is the honest zero-line: it should be well above the 10% PS bar and
    # monotonically worse with duration on a curvy test route.
    assert meds[30] > 5.0
    assert meds[180] > meds[30]
