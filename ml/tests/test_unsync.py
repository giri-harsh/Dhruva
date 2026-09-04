"""Tests for the unsynchronised pre-training corpus loader + GNSS labels +
phone-only frame alignment. Skipped if the unsynchronised S-Dataset isn't
materialised.
"""
import os
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_UNSYNC = (Path(os.environ.get("IOVNBD_ROOT", _REPO / "data" / "raw" / "IO-VNBD"))
           / "Unsynchronised V and S Dataset" / "Uncategorised IOVNB (V and S) Dataset" / "S-Dataset")


@pytest.fixture(scope="module")
def unsync_seqs():
    if not _UNSYNC.is_dir() or not any(_UNSYNC.glob("S-*.csv")):
        pytest.skip("unsynchronised S-Dataset not materialised")
    from anchor.data.unsync import discover_unsync_phone
    return discover_unsync_phone(_REPO / "data" / "raw" / "IO-VNBD")


def test_pretrain_corpus_excludes_held_out_families(unsync_seqs):
    """LEAKAGE: no Vta (test_id), Vtb (val), Driver A (S*/A*), or Driver D (Y*)
    sequence may be role='pretrain' or 'test_ood_region'."""
    forbidden = {"vta", "vtb", "s", "a", "y", "vfa"}
    for s in unsync_seqs:
        if s.role in ("pretrain", "test_ood_region"):
            assert s.family not in forbidden, (
                f"{s.seq_id} (family {s.family}) leaked into {s.role}"
            )


def test_ood_region_holdout_is_deterministic_and_ood(unsync_seqs):
    ood = [s for s in unsync_seqs if s.role == "test_ood_region"]
    assert ood, "expected some held-out OOD-region sequences"
    assert all(s.region in ("france", "nigeria") for s in ood)
    # re-discovery gives the same partition
    from anchor.data.unsync import discover_unsync_phone
    again = discover_unsync_phone(_REPO / "data" / "raw" / "IO-VNBD")
    assert {s.seq_id for s in ood} == {s.seq_id for s in again if s.role == "test_ood_region"}


def test_reduced_18col_files_load_with_six_model_channels(unsync_seqs):
    """The France S-T* files use the 18-col schema; all 6 model channels present."""
    france = [s for s in unsync_seqs if s.region == "france" and s.n_rows > 500]
    assert france
    d = france[0].df
    for c in ["accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
              "gyro_yaw_radps", "gyro_pitch_radps", "gyro_roll_radps",
              "gravity_z_mps2", "gps_speed_mps"]:
        assert c in d.columns and np.isfinite(d[c].to_numpy()).mean() > 0.9


def test_gnss_labels_reject_poor_accuracy(unsync_seqs):
    from anchor.data.gnss_labels import GnssSpeedLabeller
    s = max((x for x in unsync_seqs if x.role == "pretrain"), key=lambda x: x.n_rows)
    lab = GnssSpeedLabeller(s)
    oks = [lab.label(i, i + 20) for i in range(0, s.n_rows - 20, 200)]
    assert any(w.ok for w in oks)
    for w in oks:
        if w.ok:
            assert 1.0 <= w.label_sigma_mps <= 15.0    # weak-label range


def test_phone_only_alignment_runs(unsync_seqs):
    from anchor.data.features import align_phone_only, phone_df_model_features
    s = next(x for x in unsync_seqs if x.role == "pretrain" and x.n_rows > 3000)
    al = align_phone_only(s.df)
    feats = phone_df_model_features(s.df, al)
    assert feats.shape == (s.n_rows, 6)
    R = al["R"]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)   # a proper rotation
