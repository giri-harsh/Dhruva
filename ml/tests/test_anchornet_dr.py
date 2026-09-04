"""Plumbing tests for the standalone ANCHOR-Net dead-reckoner. Uses a
randomly-initialised model (no training needed) — asserts shapes, the FR-24
bounds path, and that the online bias estimator only looks pre-outage.
"""
import numpy as np
import pytest
import torch

from anchor.models.anchornet import AnchorNet
from anchor.splits.protocol import assign_all


@pytest.fixture(scope="module")
def dr_and_seq(sequences, tmp_path_factory):
    from anchor.eval.anchornet_dr import AnchorNetDeadReckoner

    d = tmp_path_factory.mktemp("dr")
    ckpt = d / "anchornet_seed0.pt"
    torch.manual_seed(0)
    torch.save(AnchorNet().state_dict(), ckpt)
    norm = str((__import__("pathlib").Path(__file__).resolve().parents[2]
                / "ml" / "splits" / "normalizer_train.json"))
    dr = AnchorNetDeadReckoner(checkpoint_path=str(ckpt), normalizer_path=norm)
    seq = next(s for s in assign_all(sequences)["test_id"]
               if s.n_rows > 2500 and s.meta["n_segments"] == 1)
    return dr, seq


def _outage(seq, start, dur_s):
    from anchor.eval.outages import OutageSpec
    return OutageSpec(seq.seq_id, 0, start, dur_s)


def test_predict_outage_shapes(dr_and_seq):
    dr, seq = dr_and_seq
    o = _outage(seq, 800, 60)
    e, n, hdg = dr.predict_outage(seq, o)
    assert len(e) == len(n) == o.n_rows + 1
    assert np.all(np.isfinite(e)) and np.all(np.isfinite(n))
    assert np.isfinite(hdg)


def test_fr24_rejects_implausible_speed(dr_and_seq, monkeypatch):
    dr, seq = dr_and_seq
    from anchor.eval import anchornet_dr as mod

    class Berserk(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return {"velocity_mean_mps": torch.full((b, 1), 9_999.0),
                    "velocity_log_variance": torch.zeros(b, 1)}

    dr._net = Berserk()
    dr.rejected_windows = 0
    dr.online_bias = False
    dr.predict_outage(seq, _outage(seq, 800, 60))
    assert dr.rejected_windows > 0


def test_online_bias_is_finite_and_bounded(dr_and_seq):
    dr, seq = dr_and_seq
    feats = dr._features(seq)
    b = dr._estimate_speed_bias(seq, feats, 1500)
    assert np.isfinite(b) and -8.0 <= b <= 8.0


def test_online_bias_ignores_post_outage_corruption(dr_and_seq):
    """Corrupting rows at/after outage_start must not move the estimate."""
    dr, seq = dr_and_seq
    feats = dr._features(seq).copy()
    a = 1500
    b1 = dr._estimate_speed_bias_from_arrays(feats, seq.df["veh_speed_mps"].to_numpy(), a)
    gt = seq.df["veh_speed_mps"].to_numpy().copy()
    gt[a:] = -999.0
    feats[a:] = 0.0
    b2 = dr._estimate_speed_bias_from_arrays(feats, gt, a)
    assert abs(b1 - b2) < 1e-6
