"""FR-31 integrity bench CLI.

    python -m anchor.integrity.run_bench [--iovnbd-root PATH] [--seed N]
        [--detector innovation]   # 'chisquare' once Kamal's gate is wired in

Writes ml/eval/integrity_roc.json — the dashboard reads it, and CI checks the
committed expected curve (PRD FR-31: "reproducible from a fixed seed and checked
against a committed expected curve").
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..data.sync import discover_sequences
from ..splits.protocol import assign_all
from .roc import InnovationResidualDetector, score_detector

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IOVNBD = os.environ.get("IOVNBD_ROOT", str(_ROOT / "data" / "raw" / "IO-VNBD"))
_OUT = _ROOT / "ml" / "eval" / "integrity_roc.json"

FAMILIES = {
    "step":      [5.0, 10.0, 20.0, 40.0, 80.0],           # metres
    "drag":      [0.1, 0.25, 0.5, 1.0, 2.0],              # m/s walk-off rate
    "jam":       [5.0, 15.0, 30.0, 60.0],                 # seconds of outage
    "multipath": [0.1, 0.25, 0.5, 0.9],                   # affected fraction
}

_DETECTORS = {"innovation": InnovationResidualDetector}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--detector", default="innovation", choices=list(_DETECTORS))
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    seqs = discover_sequences(args.iovnbd_root)
    splits = assign_all(seqs)
    # attack the held-out test route; measure false-rejection on the OOD-driver
    # clean sequences (independent, never attacked)
    attack_seqs = [s for s in splits["test_id"] if s.meta["n_segments"] and s.n_rows > 60 * 10]
    clean_seqs = [s for s in splits["test_ood_driver"] if s.n_rows > 60 * 10]

    result = score_detector(
        _DETECTORS[args.detector](), clean_seqs, attack_seqs,
        families=FAMILIES, seed=args.seed,
    )
    result["provenance"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "attack_sequences": [s.seq_id for s in attack_seqs],
        "clean_sequences": [s.seq_id for s in clean_seqs],
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    for fam, r in result["families"].items():
        print(f"  {fam:10s} operating_thr={result['operating_threshold']}  "
              f"provably-undetected <= {r['provably_undetected_param']}  "
              f"det@op={ {k: r['detection_at_operating'][k] for k in list(r['detection_at_operating'])[:3]} }")


if __name__ == "__main__":
    main()
