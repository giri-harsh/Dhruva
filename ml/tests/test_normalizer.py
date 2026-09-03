import json

import numpy as np
import pytest

from anchor.contract import NUM_FEATURES
from anchor.splits.normalizer import Normalizer


def _fake_windows(n, loc, scale, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, size=(n, 20, NUM_FEATURES))


def test_fit_transform_roundtrip():
    x = _fake_windows(500, loc=3.0, scale=2.0, seed=0)
    norm = Normalizer.fit(x, fit_on="test")
    y = norm.transform(x)
    assert np.allclose(y.mean(axis=(0, 1)), 0.0, atol=1e-4)
    assert np.allclose(y.std(axis=(0, 1)), 1.0, atol=1e-3)


def test_save_load_roundtrip(tmp_path):
    x = _fake_windows(100, 1.0, 1.0, 1)
    norm = Normalizer.fit(x, fit_on="test")
    p = tmp_path / "n.json"
    norm.save(p)
    back = Normalizer.load(p)
    assert np.allclose(norm.mean, back.mean)
    assert np.allclose(norm.std, back.std)


def test_normaliser_fitted_on_train_only():
    """PRD §6.2 hygiene rule 2: the normaliser sees TRAIN windows only. A
    normaliser fitted on train must NOT already be normalised w.r.t. a
    differently-distributed val/test set — proving no test data leaked into the
    fit."""
    train = _fake_windows(2000, loc=0.0, scale=1.0, seed=10)
    test = _fake_windows(2000, loc=5.0, scale=3.0, seed=11)  # different distribution

    norm = Normalizer.fit(train, fit_on="split=train")

    train_n = norm.transform(train)
    test_n = norm.transform(test)

    # train is standardised by its own stats
    assert np.allclose(train_n.mean(axis=(0, 1)), 0.0, atol=0.05)
    assert np.allclose(train_n.std(axis=(0, 1)), 1.0, atol=0.05)
    # test, run through the TRAIN normaliser, is visibly off-centre and off-scale
    # (it would be ~0/~1 if its own statistics had contaminated the fit)
    assert np.abs(test_n.mean()) > 2.0
    assert test_n.std() > 2.0


@pytest.mark.slow
@pytest.mark.usefixtures("iovnbd_root")
def test_committed_normalizer_matches_a_fresh_train_fit(iovnbd_root, tmp_path):
    """The committed ml/splits/normalizer_train.json reproduces from a fresh
    build (same guard against a stale artefact as contracts-ci)."""
    from anchor.splits.build_manifests import build
    from pathlib import Path

    committed = json.loads(
        (Path(__file__).resolve().parents[2] / "ml" / "splits" / "normalizer_train.json")
        .read_text()
    )
    # rebuild into a temp location by monkeypatching _OUT is overkill; instead
    # just refit from the same code path and compare stats.
    from anchor.splits import build_manifests as bm
    import numpy as np
    from anchor.data.sync import discover_sequences
    from anchor.data.features import align_sequence_to_vehicle_frame, sequence_model_features
    from anchor.splits.protocol import assign_all
    from anchor.splits.windower import SequenceWindower
    from anchor.splits.normalizer import Normalizer

    seqs = discover_sequences(iovnbd_root)
    train_seqs = assign_all(seqs)["train"]
    w = SequenceWindower(training=True)
    stack = []
    for seq in train_seqs:
        feats = sequence_model_features(seq, align_sequence_to_vehicle_frame(seq))
        for win in w.windows(seq):
            stack.append(feats[win.start:win.stop])
    fresh = Normalizer.fit(np.stack(stack), fit_on="test-refit")

    assert np.allclose(fresh.mean, committed["mean"], atol=1e-6), "normalizer_train.json is stale"
    assert np.allclose(fresh.std, committed["std"], atol=1e-6), "normalizer_train.json is stale"
