import numpy as np
import pytest

from anchor.data.labels import SequenceLabeller, fit_wheel_radius
from anchor.splits.protocol import ALL_SPLITS, assign_all


def test_split_partition_is_disjoint_and_total(sequences):
    splits = assign_all(sequences)
    assert set(splits) == set(ALL_SPLITS)
    seen = {}
    for name, seqs in splits.items():
        for s in seqs:
            assert s.seq_id not in seen, f"{s.seq_id} in both {seen.get(s.seq_id)} and {name}"
            seen[s.seq_id] = name
    assert len(seen) == len(sequences)


def test_route_family_never_spans_train_and_a_held_out_split(sequences):
    """The real leakage invariant: no route family may have clips in `train`
    AND in any evaluation split. A family may span several held-out splits
    (e.g. Vtb -> val + test_repeat_corridor via the vtb05 override) — that
    leaks nothing into training."""
    splits = assign_all(sequences)
    fam_to_splits: dict[str, set[str]] = {}
    for name, seqs in splits.items():
        for s in seqs:
            fam_to_splits.setdefault(s.route_family, set()).add(name)
    eval_splits = {"test_id", "test_ood_driver", "test_repeat_corridor"}
    for fam, names in fam_to_splits.items():
        if "train" in names:
            assert not (names & eval_splits), f"family {fam} spans train and {names & eval_splits}"


def test_repeat_corridor_pair_is_held_out(sequences):
    splits = assign_all(sequences)
    train_ids = {s.seq_id for s in splits["train"]}
    assert "vfa02" not in train_ids and "vtb05" not in train_ids


def test_driver_A_fully_held_out(sequences):
    splits = assign_all(sequences)
    for name, seqs in splits.items():
        drivers = {s.driver for s in seqs}
        if name == "test_ood_driver":
            assert drivers == {"A"}
        elif name in ("train", "val"):
            assert "A" not in drivers, f"Driver A leaked into {name}"


def test_wheel_radius_is_physically_plausible(sequences):
    train = assign_all(sequences)["train"]
    fit = fit_wheel_radius(train)
    assert 0.25 < fit.radius_m < 0.36, f"implausible wheel radius {fit.radius_m}"
    assert fit.spread_m < 0.01, "per-sequence radius spread too large for one vehicle"


def test_label_sigma_includes_gnss_and_wheelcan_terms(sequences):
    train = assign_all(sequences)["train"]
    fit = fit_wheel_radius(train)
    seq = next(s for s in train if s.meta["usability"] == "use" and s.n_rows > 3000)
    lab = SequenceLabeller(seq, fit.radius_m)
    # a mid-sequence moving window
    wl = lab.label(1500, 1520)
    assert wl.parts["sigma_gnss_m"] > 0.0, "GNSS residual term must be non-zero"
    assert wl.parts["sigma_wheelcan_m"] >= 0.0
    assert wl.parts["seq_gnss_speed_rmse_mps"] > 0.0
    # sigma must be at least the GNSS floor for the window duration
    assert wl.label_sigma_m >= wl.parts["sigma_gnss_m"] * 0.9
    # displacement and mean speed are consistent
    assert wl.mean_speed_mps == pytest.approx(wl.displacement_m / 2.0, rel=1e-6)


def test_label_sigma_grows_for_weak_sequences(sequences):
    train = assign_all(sequences)["train"]
    fit = fit_wheel_radius(train)
    use_seq = next(s for s in train if s.meta["usability"] == "use" and s.n_rows > 3000)
    l_use = SequenceLabeller(use_seq, fit.radius_m).label(1500, 1520)
    # simulate a weak sequence by overriding usability
    weak_seq = use_seq
    weak_seq.meta = {**use_seq.meta, "usability": "weak"}
    l_weak = SequenceLabeller(weak_seq, fit.radius_m).label(1500, 1520)
    assert l_weak.label_sigma_m > l_use.label_sigma_m
