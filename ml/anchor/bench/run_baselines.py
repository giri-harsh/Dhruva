"""One command, every runnable baseline, one versioned JSON (PRD §6.3).

    python -m anchor.bench.run_baselines [--split test_id] [--seed 20260903]
                                         [--iovnbd-root PATH] [--out PATH]

Output: ml/bench/results/baselines_<split>_<seed>.json — carries the full
provenance block (split manifest sha, dataset shas, seed, git commit, contract
version) so a number can always be traced (PRD §14.7 rule 4). B2/B3 rows appear
here automatically once Kamal's reference filter exposes a per-outage trajectory.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..contract import CONTRACT_VERSION
from ..data.sync import discover_sequences
from ..eval.anchornet_dr import AnchorNetDeadReckoner
from ..eval.metrics import aggregate, score_outage
from ..eval.outages import sample_outages
from ..splits.protocol import assign_all
from .baselines import ALL_BASELINES, truth_enu

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IOVNBD = os.environ.get("IOVNBD_ROOT", str(_ROOT / "data" / "raw" / "IO-VNBD"))
_OUT_DIR = _ROOT / "ml" / "bench" / "results"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _provenance(split: str, seed: int, seqs) -> dict:
    splits_json = json.loads((_ROOT / "ml" / "splits" / "splits.json").read_text())
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "contract_version": CONTRACT_VERSION,
        "split": split,
        "seed": seed,
        "train_manifest_sha256": splits_json["train_manifest_sha256"],
        "n_sequences_in_split": len(seqs),
        "harness_version": "0.1.0",
    }


def run(split: str, seed: int, iovnbd_root: str, out_path: Path,
        anchornet_run: str | None = None) -> dict:
    all_seqs = discover_sequences(iovnbd_root)
    seqs = assign_all(all_seqs)[split]
    if not seqs:
        raise SystemExit(f"split '{split}' is empty")
    outages = sample_outages(seqs, seed=seed)
    by_id = {s.seq_id: s for s in seqs}

    entries = list(ALL_BASELINES)
    if anchornet_run:
        rd = Path(anchornet_run)
        ckpts = sorted(rd.glob("anchornet_seed*.pt"))
        if not ckpts:
            raise SystemExit(f"no anchornet_seed*.pt in {rd}")
        entries.append(AnchorNetDeadReckoner(
            checkpoint_path=str(ckpts[0]),
            normalizer_path=str(_ROOT / "ml" / "splits" / "normalizer_train.json"),
        ))

    results: dict[str, dict] = {}
    for bl in entries:
        if not bl.runnable:
            results[bl.id] = {"name": bl.name, "status": "cited-only"}
            continue
        scores = []
        for o in outages:
            seq = by_id[o.seq_id]
            te, tn, thdg = truth_enu(seq, o)
            pe, pn, phdg = bl.predict_outage(seq, o)
            m = min(len(pe), len(te))
            scores.append(score_outage(
                seq_id=o.seq_id, duration_s=o.duration_s, scenario=o.scenario,
                pred_e=pe[:m], pred_n=pn[:m], truth_e=te[:m], truth_n=tn[:m],
                pred_heading_end_rad=phdg, truth_heading_end_rad=thdg,
            ))
        results[bl.id] = {
            "name": bl.name,
            "status": "ok",
            **({"fr24_rejected_windows": bl.rejected_windows}
               if hasattr(bl, "rejected_windows") else {}),
            "aggregate": aggregate(scores),
            "per_outage": [s.as_dict() for s in scores],
        }

    doc = {
        "provenance": _provenance(split, seed, seqs),
        "n_outages": len(outages),
        "baselines": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    ap.add_argument("--anchornet-run", default=None,
                    help="a ml/train/runs/<name> dir — adds the ANCHOR-Net DR row")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else _OUT_DIR / f"baselines_{args.split}_{args.seed}.json"
    doc = run(args.split, args.seed, args.iovnbd_root, out, args.anchornet_run)
    print(f"wrote {out}  ({doc['n_outages']} outages)")
    for bid, r in doc["baselines"].items():
        if r["status"] != "ok":
            print(f"  {bid}: {r['status']}")
            continue
        bd = r["aggregate"]["by_duration"]
        line = "  ".join(
            f"{d}s drift={bd[d]['drift_pct']['median']}%" for d in bd if bd[d]['drift_pct']
        )
        print(f"  {bid}: {line}")


if __name__ == "__main__":
    main()
