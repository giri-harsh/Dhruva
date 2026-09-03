"""Build the committed split artefacts under ml/splits/.

    python -m anchor.splits.build_manifests [--iovnbd-root PATH]

Writes (all COMMITTED — PRD §6.2 "split manifests are committed files, not code
that regenerates them"; this script regenerates them but the JSON is the record):

  ml/splits/splits.json            whole-sequence assignment + window counts +
                                   wheel-radius fit + dataset SHA-256s
  ml/splits/normalizer_train.json  per-channel mean/std, fitted on TRAIN windows
  ml/splits/repeat_route_pairs.json corridor-overlap pairs for Kamal's FR-30
  ml/splits/README.md              what these splits can and cannot claim
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..data.features import align_sequence_to_vehicle_frame, sequence_model_features
from ..data.labels import fit_wheel_radius
from ..data.sync import discover_sequences
from .normalizer import Normalizer
from .protocol import assign_all
from .windower import SequenceWindower

_ROOT = Path(__file__).resolve().parents[3]
_OUT = _ROOT / "ml" / "splits"
_DEFAULT_IOVNBD = os.environ.get("IOVNBD_ROOT", str(_ROOT / "data" / "raw" / "IO-VNBD"))


def _manifest_sha(records) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r['seq_id']}:{r['s_sha256']}:{r['v_sha256']}".encode())
    return h.hexdigest()


def build(iovnbd_root: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    seqs = discover_sequences(iovnbd_root)
    by_id = {s.seq_id: s for s in seqs}
    splits = assign_all(seqs)

    train_win = SequenceWindower(training=True)
    infer_win = SequenceWindower(training=False)

    def rec(seq):
        m = seq.meta
        return {
            "seq_id": seq.seq_id,
            "route_family": seq.route_family,
            "driver": seq.driver,
            "region": seq.region,
            "n_rows": seq.n_rows,
            "duration_s": round(seq.duration_s, 1),
            "n_segments": m["n_segments"],
            "usability": m["usability"],
            "vib_speed_corr": m["vib_speed_corr"],
            "lsq_yaw_r2": m["lsq_yaw_r2"],
            "n_windows_train_stride": train_win.count(seq),
            "n_windows_infer_stride": infer_win.count(seq),
            "s_sha256": _sha_of(iovnbd_root, m["s_path"]),
            "v_sha256": _sha_of(iovnbd_root, m["v_path"]),
        }

    split_records = {name: [rec(s) for s in sorted(seqs_, key=lambda x: x.seq_id)]
                     for name, seqs_ in splits.items()}

    # --- wheel radius: TRAIN only ---
    train_seqs = splits["train"]
    radius = fit_wheel_radius(train_seqs)

    # --- normaliser: TRAIN windows only ---
    feat_stack = []
    for seq in train_seqs:
        align = align_sequence_to_vehicle_frame(seq)
        feats = sequence_model_features(seq, align)
        for w in train_win.windows(seq):
            feat_stack.append(feats[w.start:w.stop])
    if not feat_stack:
        raise RuntimeError("no training windows produced")
    feat_arr = np.stack(feat_stack)  # [N, 20, 6]
    manifest_sha = _manifest_sha(split_records["train"])
    norm = Normalizer.fit(feat_arr, fit_on=f"split=train n_seq={len(train_seqs)} "
                                            f"manifest_sha={manifest_sha[:12]}")
    norm.save(_OUT / "normalizer_train.json")

    splits_json = {
        "protocol": "whole-sequence holdout, split by route family; "
                    "guard band 10 s at every segment boundary; "
                    "train stride 0.5 s, inference stride 2.0 s (no overlap)",
        "iovnbd_source": "github.com/onyekpeu/IO-VNBD @ master, synchronised/Categorised",
        "contract_version_at_build": _contract_version(),
        "train_manifest_sha256": manifest_sha,
        "wheel_radius": {
            "radius_m": round(radius.radius_m, 5),
            "fit_samples": radius.n_samples,
            "per_sequence_spread_m": round(radius.spread_m, 5),
            "per_sequence_m": {k: round(v, 5) for k, v in sorted(radius.per_sequence_m.items())},
            "method": "origin regression of VBOX speed on mean wheel angular rate, "
                      "clean straight stretches only (|yaw|<0.03, v>5 m/s, sats>=4)",
        },
        "counts": {name: {
            "n_sequences": len(recs),
            "hours": round(sum(r["duration_s"] for r in recs) / 3600, 2),
            "hours_use": round(sum(r["duration_s"] for r in recs if r["usability"] == "use") / 3600, 2),
            "train_windows": sum(r["n_windows_train_stride"] for r in recs),
        } for name, recs in split_records.items()},
        "splits": split_records,
    }
    (_OUT / "splits.json").write_text(json.dumps(splits_json, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")

    _write_repeat_pairs(by_id)
    _write_readme(splits_json)
    print("wrote ml/splits/{splits.json, normalizer_train.json, repeat_route_pairs.json, README.md}")
    for name, c in splits_json["counts"].items():
        print(f"  {name:22s} {c['n_sequences']:3d} seq  {c['hours']:5.2f} h  "
              f"({c['hours_use']:.2f} h use)  {c['train_windows']:>7d} train windows")
    print(f"  wheel radius = {radius.radius_m:.4f} m  (spread {radius.spread_m:.4f} m "
          f"over {len(radius.per_sequence_m)} seqs)")


def _sha_of(root: str, rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = Path(root) / p
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _contract_version() -> str:
    from ..contract import CONTRACT_VERSION
    return CONTRACT_VERSION


def _write_repeat_pairs(by_id: dict) -> None:
    pairs = {
        "purpose": "Kamal's magnetic route memory (FR-30) needs held-out sequence "
                   "pairs that traverse the same road. IO-VNBD's SYNCHRONISED set has "
                   "no clean 'same route driven twice' pair; the entries below are "
                   "the best available: long drives that share a road CORRIDOR. All "
                   "listed sequences are in a held-out split (never train), so the "
                   "velocity model has not seen the geometry.",
        "grid_cell_m": 500,
        "pairs": [
            {
                "a": "vfa02", "b": "vtb05",
                "shared_cells_approx": 147,
                "note": "both Driver E; overlapping northbound motorway corridor "
                        "(~70 km of shared 500 m cells). vfa02 -> test_repeat_corridor, "
                        "vtb05 -> test_repeat_corridor.",
            },
        ],
        "corridor_groups": [
            {
                "seq_ids": ["m", "s1", "s2", "s4", "y1"],
                "note": "Coventry-area urban road network, shared by Drivers A/B/D. "
                        "Overlap is diffuse (20-46 shared cells per pair), not a single "
                        "repeated route. m -> train; s1/s2/s4 -> test_ood_driver; "
                        "y1 -> excluded. Use for cross-driver magnetic tests only with "
                        "that caveat.",
            },
        ],
        "recommendation": "For a proper route-repeat magnetic test, pull the "
                          "UNSYNCHRONISED Vw/Vta families (not in this checkout) and "
                          "look for same-day repeated passes there.",
    }
    (_OUT / "repeat_route_pairs.json").write_text(json.dumps(pairs, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")


def _fam_summary(recs) -> str:
    fams = sorted({f"{r['route_family']}({r['driver']})" for r in recs})
    return " + ".join(fams) if fams else "-"


def _write_readme(sj: dict) -> None:
    c = sj["counts"]
    s = sj["splits"]
    rows = "\n".join(
        f"| `{name}` | {_fam_summary(s[name])} | {c[name]['n_sequences']} | "
        f"{c[name]['hours']} ({c[name]['hours_use']}) | {purpose} |"
        for name, purpose in [
            ("train", "fit ANCHOR-Net"),
            ("val", "early stopping, HPs"),
            ("test_id", "**headline: unseen route**"),
            ("test_ood_driver", "unseen driver (Driver A)"),
            ("test_repeat_corridor", "Kamal FR-30 corridor overlap"),
            ("excluded", "parked / unusable / decoupled"),
        ]
    )
    md = f"""# `ml/splits/` — the leakage-safe split, and what it can honestly claim

Generated by `python -m anchor.splits.build_manifests`. The JSON files are the
record; regenerate only when the sequence set or protocol changes, and commit
the diff with a reason (PRD §6.2, §14.7 rule 4).

## Assignment (whole route families, never individual clips)

| Split | Families (driver) | Sequences | Hours (of which `use`) | Purpose |
|---|---|---|---|---|
{rows}

## Why clips are not the split unit

Driver E's `Vta`, `Vtb`, `Vw` families are each **one continuous road trip cut
into consecutive clips** (`vta02`→`vta03`→… march north along one road). Splitting
clip-wise would put the same 100 m of road, the same weather, the same tyre
state and mount angle on both sides of train/test — the exact leak in PRD §6.2
point 4. So the split unit is the whole drive / route family.

## What the headline number (`test_id`) means, and its limit

`test_id` = the entire `Vta` road trip through the Peak District (hilly, mixed
country roads) — roads that do **not** appear in `train` (`Vw` = Worcestershire
and the west, `M` = Coventry). So it is a true **unseen-route** result, on
terrain that also matches the hill-corridor persona and map region.

It is **not** an unseen-vehicle or (mostly) unseen-driver result: IO-VNBD is one
instrumented car, and Driver E is in both `train` and `test_id`. The only
unseen-**driver** signal available is `test_ood_driver` (all of Driver A, never
trained) — ~3 h of `use`-grade data. Driver B is a single drive (in train);
Driver D (`y1`) is unusable (phone stream decoupled from the vehicle,
`sync_speed_corr` 0.08) and is in `excluded`.

Report `test_id` first, `test_ood_driver` beside it labelled as the thinner
signal, and France/Nigeria OOD (from the unsynchronised phone tree, GNSS-label
only) separately — per PRD §6.2 / §6.7.

## Hygiene rules enforced in code (with the tests that prove it)

- no window crosses a sequence **or internal segment** boundary —
  `ml/tests/test_windower.py::test_no_cross_sequence_windows`
- 10 s guard band dropped at every segment boundary — `windower.GUARD_BAND_S`
- normaliser fitted on `train` windows only —
  `ml/tests/test_normalizer.py::test_normaliser_fitted_on_train_only`
- wheel radius / scale fitted on `train` only — `labels.fit_wheel_radius`
"""
    (_OUT / "README.md").write_text(md, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    build(ap.parse_args().iovnbd_root)


if __name__ == "__main__":
    main()
