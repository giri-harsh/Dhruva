"""CI regression gate on the golden set's 10-segment public subset (PRD §14.7
rule 3): run a model's dead-reckoner over the public segments, compare median
drift-% to a committed baseline; a > 5% relative regression fails the build.

    python -m anchor.golden.regression_gate --run ml/train/runs/<name> [--update-baseline]

The other 30 segments are held for the two permitted full evaluations and are
NOT touched here.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..bench.baselines import truth_enu
from ..data.sync import discover_sequences
from ..eval.anchornet_dr import AnchorNetDeadReckoner
from ..eval.metrics import score_outage
from ..eval.outages import OutageSpec
from ..splits.protocol import assign_all

_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = _ROOT / "ml" / "golden" / "manifest.json"
_BASELINE = _ROOT / "ml" / "golden" / "public_baseline.json"
_DEFAULT_IOVNBD = os.environ.get("IOVNBD_ROOT", str(_ROOT / "data" / "raw" / "IO-VNBD"))
REGRESSION_LIMIT_REL = 0.05


def _public_specs() -> list[OutageSpec]:
    m = json.loads(_MANIFEST.read_text())
    return [OutageSpec(s["seq_id"], s["seg_index"], s["start_row"], s["duration_s"],
                       scenario=s["scenario"])
            for s in m["segments"] if s["public_subset"]]


def run(run_dir: str, iovnbd_root: str) -> dict:
    specs = _public_specs()
    seqs = discover_sequences(iovnbd_root)
    splits = assign_all(seqs)
    by_id = {s.seq_id: s
             for name in ("test_id", "test_ood_driver", "test_repeat_corridor")
             for s in splits[name]}

    ckpt = sorted(Path(run_dir).glob("anchornet_seed*.pt"))[0]
    dr = AnchorNetDeadReckoner(
        checkpoint_path=str(ckpt),
        normalizer_path=str(_ROOT / "ml" / "splits" / "normalizer_train.json"))

    drifts = []
    per_seg = []
    for o in specs:
        seq = by_id[o.seq_id]
        te, tn, _ = truth_enu(seq, o)
        pe, pn, _ = dr.predict_outage(seq, o)
        m = min(len(pe), len(te))
        sc = score_outage(seq_id=o.seq_id, duration_s=o.duration_s, scenario=o.scenario,
                          pred_e=pe[:m], pred_n=pn[:m], truth_e=te[:m], truth_n=tn[:m])
        drifts.append(sc.drift_pct)
        per_seg.append({"key": o.key(), "scenario": o.scenario, "drift_pct": round(sc.drift_pct, 3)})

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": Path(run_dir).name,
        "checkpoint": ckpt.name,
        "median_drift_pct": round(float(np.median(drifts)), 4),
        "mean_drift_pct": round(float(np.mean(drifts)), 4),
        "per_segment": per_seg,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    run_dir = args.run if Path(args.run).is_absolute() else str(_ROOT / args.run)
    result = run(run_dir, args.iovnbd_root)
    print(f"public-subset median drift: {result['median_drift_pct']}%  "
          f"(mean {result['mean_drift_pct']}%)")

    if args.update_baseline or not _BASELINE.exists():
        _BASELINE.write_text(json.dumps(result, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"wrote baseline {_BASELINE}")
        return

    base = json.loads(_BASELINE.read_text())
    b, cur = base["median_drift_pct"], result["median_drift_pct"]
    rel = (cur - b) / b if b else 0.0
    print(f"baseline {b}%  ->  current {cur}%   ({rel:+.1%})")
    if rel > REGRESSION_LIMIT_REL:
        raise SystemExit(f"REGRESSION: median drift up {rel:.1%} (> {REGRESSION_LIMIT_REL:.0%}) "
                         f"vs the committed golden baseline")
    print("OK — within tolerance")


if __name__ == "__main__":
    main()
