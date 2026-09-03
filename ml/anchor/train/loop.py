"""ANCHOR-Net training loop (PRD §6.6).

AdamW lr 3e-4, cosine decay, weight decay 1e-4, batch 256 with speed-decile
re-weighting, early stopping on validation NLL (patience 15). Every reported
number is mean +/- std over 5 seeds — a single-seed result is never reported.

Fully reproducible from a committed config: seeds, split manifest sha, dataset
shas, and git commit are written into every metrics file.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..models.anchornet import AnchorNet, AnchorNetConfig
from .augment import Augmenter
from .dataset import AnchorWindowDataset
from .losses import context_loss, speed_loss, yaw_loss

_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 120
    warmup_epochs: int = 8            # pure-MSE phase before beta-NLL (probe: NLL-from-scratch is degenerate)
    patience: int = 15
    lambda_context: float = 0.2
    lambda_yaw: float = 0.2
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    num_workers: int = 0
    num_threads: int = 5             # this box thrashes at torch's default 12
    model: AnchorNetConfig = field(default_factory=AnchorNetConfig)


def _seed_everything(s: int, threads: int = 5):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(threads)


def _run_epoch(model, loader, opt, cfg, train: bool, device, *, use_nll: bool):
    model.train(train)
    agg = {"loss": 0.0, "nll": 0.0, "rmse": 0.0, "bias": 0.0, "sigma": 0.0, "n": 0}
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["target_speed"].to(device)
        ls = batch["label_sigma"].to(device)
        w = batch.get("mean_weight")
        w = w.to(device) if w is not None else None
        with torch.set_grad_enabled(train):
            out = model(x)
            loss, stats = speed_loss(
                out["velocity_mean_mps"], out["velocity_log_variance"], y, ls,
                use_nll=use_nll, sample_weight=w)
            if "context_logits" in out and "context_label" in batch:
                loss = loss + cfg.lambda_context * context_loss(
                    out["context_logits"], batch["context_label"].to(device),
                    batch["context_mask"].to(device))
            if "yaw_increment_rad" in out and "yaw_target" in batch:
                loss = loss + cfg.lambda_yaw * yaw_loss(
                    out["yaw_increment_rad"], out["yaw_log_variance"], batch["yaw_target"].to(device))
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        bs = x.size(0)
        for k in ("loss",):
            agg[k] += float(loss) * bs
        agg["nll"] += stats["nll"] * bs
        agg["rmse"] += stats["rmse"] * bs
        agg["bias"] += stats["bias"] * bs
        agg["sigma"] += stats["pred_sigma_mean"] * bs
        agg["n"] += bs
    n = max(agg["n"], 1)
    return {k: agg[k] / n for k in ("loss", "nll", "rmse", "bias", "sigma")}


def train_one_seed(seed, train_seqs, val_seqs, *, radius_m, normalizer, cfg: TrainConfig,
                   out_dir: Path, device="cpu") -> dict:
    _seed_everything(seed, cfg.num_threads)
    aug = Augmenter()
    ds_tr = AnchorWindowDataset(train_seqs, radius_m=radius_m, normalizer=normalizer,
                                training=True, augmenter=aug, seed=seed)
    ds_va = AnchorWindowDataset(val_seqs, radius_m=radius_m, normalizer=normalizer,
                                training=False, seed=seed)
    sampler = WeightedRandomSampler(ds_tr.sample_weights().tolist(), len(ds_tr), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, sampler=sampler,
                       num_workers=cfg.num_workers, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    model = AnchorNet(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    print(f"  seed {seed}: {len(ds_tr)} train / {len(ds_va)} val windows, "
          f"{model.num_parameters()} params, warmup {cfg.warmup_epochs} (MSE) then beta-NLL",
          flush=True)
    # selection metric is val RMSE — that is the deliverable. Only start
    # tracking "best" once beta-NLL is on (post-warmup) so calibration is real.
    best = {"val_rmse": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(cfg.max_epochs):
        use_nll = epoch >= cfg.warmup_epochs
        t0 = time.time()
        tr = _run_epoch(model, dl_tr, opt, cfg, True, device, use_nll=use_nll)
        va = _run_epoch(model, dl_va, opt, cfg, False, device, use_nll=use_nll)
        sched.step()
        dt = round(time.time() - t0, 1)
        history.append({"epoch": epoch, "phase": "nll" if use_nll else "mse",
                        "train": tr, "val": va, "s": dt})
        improved = va["rmse"] < best["val_rmse"] - 1e-3
        if improved and (use_nll or epoch >= cfg.warmup_epochs - 2):
            best = {"val_rmse": va["rmse"], "val_nll": va["nll"], "epoch": epoch,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        print(f"  seed {seed} e{epoch:03d} {dt:5.1f}s [{'nll' if use_nll else 'mse'}]  "
              f"tr_rmse={tr['rmse']:.3f} val_rmse={va['rmse']:.3f} "
              f"val_bias={va['bias']:+.2f} sigma={va['sigma']:.2f}"
              f"{'  *' if improved else ''}", flush=True)
        if best["epoch"] >= 0 and epoch - best["epoch"] >= cfg.patience:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    if best["state"] is not None:
        torch.save(best["state"], out_dir / f"anchornet_seed{seed}.pt")
    result = {
        "seed": seed,
        "n_params": model.num_parameters(),
        "best_epoch": best["epoch"],
        "best_val_nll": best.get("val_nll"),
        "best_val_rmse": best["val_rmse"] if best["epoch"] >= 0 else None,
        "epochs_run": len(history),
        "history": history,
    }
    (out_dir / f"metrics_seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n",
                                                      encoding="utf-8", newline="\n")
    return result


def summarise(results: list[dict]) -> dict:
    def ms(key):
        v = np.array([r[key] for r in results if r[key] is not None], dtype=float)
        return {"mean": round(float(v.mean()), 5), "std": round(float(v.std()), 5), "n": len(v)}
    return {
        "n_params": results[0]["n_params"],
        "val_nll": ms("best_val_nll"),
        "val_rmse_mps": ms("best_val_rmse"),
        "best_epoch": ms("best_epoch"),
        "seeds": [r["seed"] for r in results],
    }
