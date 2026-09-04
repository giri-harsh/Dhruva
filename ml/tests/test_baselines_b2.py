import numpy as np
import pytest

from anchor.bench.baselines import B2Strapdown, B3Eskf


def test_b3_not_runnable_until_kamal_ships_eskf():
    assert B3Eskf().runnable is False


def test_strapdown_straight_constant_speed_tracks_true_path():
    from reference.anchor_ref import strapdown_dead_reckon
    T = 300
    feats = np.zeros((T, 6))              # no accel, no rotation -> constant velocity
    out = strapdown_dead_reckon(feats, dt_s=0.1, v0_mps=10.0, heading0_rad=0.0)
    # heading 0 = due north; after 30 s at 10 m/s -> ~300 m north, ~0 east
    assert out["north_m"][-1] == pytest.approx(300.0, rel=0.05)
    assert abs(out["east_m"][-1]) < 1.0


def test_strapdown_diverges_quadratically_on_accel_bias():
    from reference.anchor_ref import strapdown_dead_reckon
    T = 300
    feats = np.zeros((T, 6))
    feats[:, 0] = 0.2                     # 0.2 m/s^2 forward bias
    out = strapdown_dead_reckon(feats, dt_s=0.1, v0_mps=0.0, heading0_rad=0.0)
    # x = 0.5 a t^2 = 0.5 * 0.2 * 30^2 = 90 m
    assert out["north_m"][-1] == pytest.approx(90.0, rel=0.1)


@pytest.mark.usefixtures("sequences")
def test_b2_runs_on_real_outage_and_is_worse_than_b1(sequences):
    from anchor.bench.baselines import B1ConstantVelocity, truth_enu
    from anchor.eval.metrics import score_outage
    from anchor.eval.outages import sample_outages
    from anchor.splits.protocol import assign_all

    test = assign_all(sequences)["test_id"]
    outs = sample_outages(test, seed=1)[:20]
    by = {s.seq_id: s for s in test}
    b1, b2 = B1ConstantVelocity(), B2Strapdown()
    d1, d2 = [], []
    for o in outs:
        s = by[o.seq_id]
        te, tn, _ = truth_enu(s, o)
        for pred, acc in ((b1, d1), (b2, d2)):
            pe, pn, _ = pred.predict_outage(s, o)
            m = min(len(pe), len(te))
            sc = score_outage(seq_id=o.seq_id, duration_s=o.duration_s, scenario="",
                              pred_e=pe[:m], pred_n=pn[:m], truth_e=te[:m], truth_n=tn[:m])
            if np.isfinite(sc.drift_pct):
                acc.append(sc.drift_pct)
    # B2 (double integration) drifts more than B1 (hold velocity) — the PRD §1.2 point
    assert np.median(d2) > np.median(d1)
