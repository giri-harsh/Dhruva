"""Train ANCHOR-Net, all seeds, from a committed config.

    python -m anchor.train.run [--iovnbd-root PATH] [--out ml/train/runs/<name>]
                               [--seeds 0,1,2,3,4] [--max-epochs N] [--smoke]

--smoke runs 1 seed / few epochs / tiny subset for a wiring check.
Checkpoints (.pt) and per-seed metrics land in --out (git-ignored); the
summary (mean +/- std over seeds) is printed and written to summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

# torch 2.5 SequentialLR calls child .step(epoch) internally -> noisy deprecation
warnings.filterwarnings("ignore", message=".*epoch parameter in.*scheduler.step.*")

from ..data.labels import fit_wheel_radius
from ..data.sync import discover_sequences
from ..splits.normalizer import Normalizer
from ..splits.protocol import assign_all
from .loop import TrainConfig, summarise, train_one_seed

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IOVNBD = os.environ.get("IOVNBD_ROOT", str(_ROOT / "data" / "raw" / "IO-VNBD"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--init-from", default=None,
                    help="a Stage-1 pretrain.pt to initialise weights from (PRD §6.6)")
    ap.add_argument("--context", action="store_true", help="enable Head C (S-15)")
    ap.add_argument("--lambda-context", type=float, default=None,
                    help="Head C loss weight; 0 = trained-but-not-backprop'd (ablation row 7)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.seeds:
        cfg.seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.max_epochs:
        cfg.max_epochs = args.max_epochs
    if args.context:
        from ..models.anchornet import AnchorNetConfig
        cfg.model = AnchorNetConfig(enable_context_head=True)
    if args.lambda_context is not None:
        cfg.lambda_context = args.lambda_context
    if args.smoke:
        cfg.seeds = (0,)
        cfg.max_epochs = 4
        cfg.warmup_epochs = 2
        cfg.patience = 4

    name = args.out or f"ml/train/runs/{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_dir = _ROOT / name if not Path(name).is_absolute() else Path(name)

    seqs = discover_sequences(args.iovnbd_root)
    splits = assign_all(seqs)
    train_seqs, val_seqs = splits["train"], splits["val"]
    if args.smoke:
        train_seqs = train_seqs[:3]
        val_seqs = val_seqs[:2]

    radius = fit_wheel_radius(splits["train"]).radius_m
    normalizer = Normalizer.load(_ROOT / "ml" / "splits" / "normalizer_train.json")

    results = []
    for seed in cfg.seeds:
        print(f"=== seed {seed} ===")
        r = train_one_seed(seed, train_seqs, val_seqs, radius_m=radius,
                           normalizer=normalizer, cfg=cfg, out_dir=out_dir,
                           init_from=args.init_from)
        print(f"  params={r['n_params']}  best_epoch={r['best_epoch']}  "
              f"val_nll={r['best_val_nll']:.4f}  val_rmse={r['best_val_rmse']:.4f} m/s")
        results.append(r)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": {
            "lr": cfg.lr, "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs, "patience": cfg.patience,
            "seeds": list(cfg.seeds), "model_trunk": cfg.model.trunk,
            "model_hidden": cfg.model.hidden,
            "init_from": args.init_from,
        },
        "wheel_radius_m": round(radius, 5),
        "summary": summarise(results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    print("\n=== summary (mean +/- std over seeds) ===")
    print(json.dumps(summary["summary"], indent=2))
    print(f"\nwrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
