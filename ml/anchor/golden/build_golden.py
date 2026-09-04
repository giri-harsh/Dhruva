"""Freeze the golden outage set (PRD §14.7).

40 outage segments from TEST splits only (test_id + test_ood_driver +
test_repeat_corridor), stratified across the scenario mix and all four
durations, checksummed, committed to ml/golden/manifest.json.

Rules bound by §14.7:
  * never used for training / HP selection / early stopping / architecture choice
  * evaluated at most twice before the internal round
  * CI runs a regression gate on a 10-segment PUBLIC subset every push
    (median drift regressing > 5% relative fails the build)
  * the manifest is APPEND-ONLY — any change needs a PR that says why

    python -m anchor.golden.build_golden [--iovnbd-root PATH] [--n 40] [--seed ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..contract import SAMPLE_RATE_HZ
from ..data.sync import discover_sequences
from ..eval.geo import lla_to_local_enu, path_length
from ..eval.outages import DURATIONS_S, sample_outages
from ..splits.protocol import assign_all

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IOVNBD = os.environ.get("IOVNBD_ROOT", str(_ROOT / "data" / "raw" / "IO-VNBD"))
_MANIFEST = _ROOT / "ml" / "golden" / "manifest.json"
_TEST_SPLITS = ("test_id", "test_ood_driver", "test_repeat_corridor")


def _seg_record(seq, o) -> dict:
    d = seq.df
    a, b = o.start_row, o.stop_row
    lat = d["veh_gt_lat_deg"].to_numpy()[a:b + 1]
    lon = d["veh_gt_lon_deg"].to_numpy()[a:b + 1]
    e, n = lla_to_local_enu(lat, lon, lat[0], lon[0])
    return {
        "key": o.key(),
        "seq_id": o.seq_id,
        "seg_index": o.seg_index,
        "start_row": o.start_row,
        "duration_s": o.duration_s,
        "scenario": o.scenario,
        "distance_m": round(path_length(e, n), 2),
        "start_lat": round(float(lat[0]), 6),
        "start_lon": round(float(lon[0]), 6),
    }


def build(iovnbd_root: str, n: int, seed: int) -> dict:
    seqs = discover_sequences(iovnbd_root)
    splits = assign_all(seqs)
    test_seqs = [s for name in _TEST_SPLITS for s in splits[name]]
    by_id = {s.seq_id: s for s in test_seqs}

    pool = sample_outages(test_seqs, seed=seed, per_duration_per_seq=3)
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)

    # stratified pick: aim for balance across (scenario, duration)
    want_per_dur = n // len(DURATIONS_S)
    chosen: list = []
    seen_keys = set()
    for dur in DURATIONS_S:
        cand = [o for o in pool if o.duration_s == dur]
        # round-robin scenarios
        by_scen: dict[str, list] = {}
        for o in cand:
            by_scen.setdefault(o.scenario, []).append(o)
        picks = []
        scen_cycle = sorted(by_scen, key=lambda k: -len(by_scen[k]))
        i = 0
        while len(picks) < want_per_dur and any(by_scen.values()):
            s = scen_cycle[i % len(scen_cycle)]
            if by_scen[s]:
                o = by_scen[s].pop()
                if o.key() not in seen_keys:
                    picks.append(o); seen_keys.add(o.key())
            i += 1
        chosen += picks
    # top up to n from the remaining pool
    for o in pool:
        if len(chosen) >= n:
            break
        if o.key() not in seen_keys:
            chosen.append(o); seen_keys.add(o.key())

    records = sorted((_seg_record(by_id[o.seq_id], o) for o in chosen),
                     key=lambda r: r["key"])
    # first 10 (deterministic by key) are the CI public subset
    for i, r in enumerate(records):
        r["public_subset"] = i < 10

    from collections import Counter
    manifest = {
        "spec": "PRD §14.7 golden outage set",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "iovnbd_source": "github.com/onyekpeu/IO-VNBD @ master, synchronised/Categorised",
        "seed": seed,
        "n_segments": len(records),
        "durations_s": list(DURATIONS_S),
        "scenario_mix": dict(Counter(r["scenario"] for r in records)),
        "by_split": dict(Counter(
            next(name for name in _TEST_SPLITS
                 if r["seq_id"] in {s.seq_id for s in splits[name]})
            for r in records)),
        "rules": [
            "never used for training / HP / early-stopping / architecture choice",
            "evaluated at most twice before the internal round",
            "CI regression gate on the 10-segment public_subset every push",
            "this manifest is APPEND-ONLY; changes need a PR explaining why",
        ],
        "content_sha256": None,
        "segments": records,
    }
    body = json.dumps(manifest["segments"], sort_keys=True).encode()
    manifest["content_sha256"] = hashlib.sha256(body).hexdigest()
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()

    if _MANIFEST.exists() and not args.force:
        raise SystemExit(f"{_MANIFEST} already exists — it is append-only. "
                         f"Use --force only with a PR explaining why.")
    m = build(args.iovnbd_root, args.n, args.seed)
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {_MANIFEST}  ({m['n_segments']} segments)")
    print(f"  scenario mix: {m['scenario_mix']}")
    print(f"  by split:     {m['by_split']}")
    print(f"  sha256:       {m['content_sha256'][:16]}")


if __name__ == "__main__":
    main()
