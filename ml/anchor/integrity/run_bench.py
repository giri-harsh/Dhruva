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
_EXPECTED = _ROOT / "ml" / "eval" / "integrity_roc_expected.json"
DET_TOL = 0.10                      # allowed drop in detection@operating vs expected

FAMILIES = {
    "step":      [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0],   # metres
    "drag":      [0.05, 0.1, 0.2, 0.4, 0.8, 1.6],                # m/s walk-off rate
    "jam":       [5.0, 15.0, 30.0, 60.0],                        # seconds of outage
    "multipath": [0.05, 0.1, 0.25, 0.5, 0.9],                    # affected fraction
}

_DETECTORS = {"innovation": InnovationResidualDetector}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--detector", default="innovation", choices=list(_DETECTORS))
    ap.add_argument("--out", default=str(_OUT))
    ap.add_argument("--check", action="store_true",
                    help="compare against ml/eval/integrity_roc_expected.json and "
                         "exit non-zero on a >10pp detection regression (CI)")
    ap.add_argument("--update-expected", action="store_true")
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
        print(f"  {fam:10s} op_thr={result['operating_threshold']}  "
              f"FR@op={r['false_rejection_at_operating']}  "
              f"provably-undetected <= {r['provably_undetected_param']}  "
              f"det@op={r['detection_at_operating']}")

    if args.update_expected or (args.check and not _EXPECTED.exists()):
        _EXPECTED.write_text(json.dumps(_expected_view(result), indent=2) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"wrote expected {_EXPECTED}")
        return
    if args.check:
        _run_check(result)


def _expected_view(result: dict) -> dict:
    return {
        "detector": result["detector"], "seed": result["seed"],
        "operating_threshold": result["operating_threshold"],
        "families": {f: {"detection_at_operating": r["detection_at_operating"],
                         "provably_undetected_param": r["provably_undetected_param"],
                         "false_rejection_at_operating": r["false_rejection_at_operating"]}
                     for f, r in result["families"].items()},
    }


def _run_check(result: dict) -> None:
    exp = json.loads(_EXPECTED.read_text())
    problems = []
    for fam, er in exp["families"].items():
        cr = result["families"].get(fam, {})
        for p, ed in er["detection_at_operating"].items():
            cd = cr.get("detection_at_operating", {}).get(p)
            if ed is not None and cd is not None and cd < ed - DET_TOL:
                problems.append(f"{fam} p={p}: detection {cd:.2f} < expected {ed:.2f} - {DET_TOL}")
        if cr.get("false_rejection_at_operating", 0) > er["false_rejection_at_operating"] + 0.03:
            problems.append(f"{fam}: false-rejection {cr['false_rejection_at_operating']} "
                            f"> expected {er['false_rejection_at_operating']} + 0.03")
    if problems:
        raise SystemExit("INTEGRITY REGRESSION:\n  " + "\n  ".join(problems))
    print("OK — integrity ROC within tolerance of the committed expected curve")


if __name__ == "__main__":
    main()
