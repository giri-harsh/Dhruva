"""Stage-1 pre-training on the unsynchronised smartphone corpus (PRD §6.6).

Weak GNSS speed labels, ~23 h (England Vw/M + France), leakage-checked (no Vta /
Vtb / Driver-A / Driver-D route in the corpus — see anchor.data.unsync).
Then `anchor.train.run --init-from <this>` fine-tunes on the synchronised
`train` split's clean wheel-speed labels.

    python -m anchor.train.pretrain [--iovnbd-root PATH] [--out ml/train/pretrain/<name>]
        [--epochs 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*epoch parameter in.*scheduler.step.*")

from ..contract import SAMPLE_RATE_HZ, WINDOW_SIZE_SAMPLES
from ..data.features import align_phone_only, phone_df_model_features
from ..data.gnss_labels import GnssSpeedLabeller
from ..data.unsync import discover_unsync_phone, summarise
from ..splits.normalizer import Normalizer
from ..splits.windower import SequenceWindower
from .augment import BatchAugmenter
from .losses import speed_loss

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IOVNBD = os.environ.get("IOVNBD_ROOT", str(_ROOT / "data" / "raw" / "IO-VNBD"))
_WIN_DUR = WINDOW_SIZE_SAMPLES / SAMPLE_RATE_HZ
_CACHE = _ROOT / "ml" / ".cache" / "features_unsync"


def _aligned(seq) -> np.ndarray:
    _CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(seq.meta["path"].encode()).hexdigest()[:16]
    p = _CACHE / f"{seq.seq_id}_{key}.npy"
    if p.exists():
        return np.load(p)
    feats = phone_df_model_features(seq.df, align_phone_only(seq.df))
    np.save(p, feats)
    return feats


def _build(seqs, windower, normalizer):
    X, y, s = [], [], []
    for seq in seqs:
        feats = _aligned(seq)
        lab = GnssSpeedLabeller(seq)
        for w in windower.windows(seq):
            fw = feats[w.start:w.stop]
            if len(fw) < WINDOW_SIZE_SAMPLES or not np.all(np.isfinite(fw)):
                continue
            wl = lab.label(w.start, w.stop)
            if not wl.ok:
                continue
            X.append(normalizer.transform(fw[None])[0])
            y.append(wl.mean_speed_mps)
            s.append(wl.label_sigma_mps)
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
            torch.tensor(np.array(s), dtype=torch.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=99,
                    help="epochs of pure MSE before beta-NLL. Default 99 = MSE-only: "
                         "pre-training just needs the vibration->speed features, not a "
                         "calibrated variance head (Stage-2 fine-tune calibrates).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(5)
    from ..models.anchornet import AnchorNet, AnchorNetConfig

    out_dir = _ROOT / (args.out or f"ml/train/pretrain/{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = discover_unsync_phone(args.iovnbd_root)
    pre = [s for s in seqs if s.role == "pretrain" and s.n_rows > 40 * SAMPLE_RATE_HZ]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pre)
    n_val = max(2, len(pre) // 6)
    val_seqs, tr_seqs = pre[:n_val], pre[n_val:]
    print(f"pretrain corpus: {summarise(seqs)}", flush=True)
    print(f"  stage-1 train {len(tr_seqs)} seq / val {len(val_seqs)} seq", flush=True)

    norm = Normalizer.load(_ROOT / "ml" / "splits" / "normalizer_train.json")
    wtr = SequenceWindower(training=True)
    wev = SequenceWindower(training=False)
    Xtr, ytr, str_ = _build(tr_seqs, wtr, norm)
    Xva, yva, _ = _build(val_seqs, wev, norm)
    print(f"  windows: train {len(ytr)}  val {len(yva)}  "
          f"predict-mean val RMSE {yva.std():.3f}  label_sigma median {str_.median():.2f}",
          flush=True)

    net = AnchorNet(AnchorNetConfig())
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    aug = BatchAugmenter()
    gen = torch.Generator().manual_seed(args.seed)
    bs = 512
    best = (1e9, -1)
    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        perm = torch.randperm(len(Xtr))
        use_nll = ep >= args.warmup
        for i in range(0, len(Xtr) - bs, bs):
            idx = perm[i:i + bs]
            xb = aug(Xtr[idx], gen)
            o = net(xb)
            loss, _ = speed_loss(o["velocity_mean_mps"], o["velocity_log_variance"],
                                 ytr[idx], str_[idx], use_nll=use_nll)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xva)["velocity_mean_mps"].squeeze(-1)
            rmse = float(torch.sqrt(torch.mean((pv - yva) ** 2)))
            bias = float((pv - yva).mean())
        star = ""
        if rmse < best[0] and ep >= 2:      # skip the first noisy epochs
            best = (rmse, ep)
            torch.save(net.state_dict(), out_dir / "pretrain.pt")
            star = "  *"
        print(f"  e{ep:03d} {time.time()-t0:5.1f}s [{'nll' if use_nll else 'mse'}] "
              f"val_rmse {rmse:.3f} bias {bias:+.2f}{star}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": summarise(seqs),
        "n_train_windows": len(ytr), "n_val_windows": len(yva),
        "best_val_rmse_mps": round(best[0], 4), "best_epoch": best[1],
        "checkpoint": "pretrain.pt",
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out_dir}/pretrain.pt  (best val RMSE {best[0]:.3f} @ e{best[1]})", flush=True)


if __name__ == "__main__":
    main()
