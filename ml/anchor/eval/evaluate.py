"""Window-level evaluation of a trained ANCHOR-Net on a split.

Distinct from the outage/trajectory bench (anchornet_dr + run_baselines): this
scores Head A / Head B directly on every inference-stride window of a split —
speed RMSE, bias, and the full FR-08 calibration report, broken down by speed
decile and by sequence usability.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..contract import SAMPLE_RATE_HZ, WINDOW_SIZE_SAMPLES
from ..data.features import align_sequence_to_vehicle_frame, sequence_model_features
from ..data.labels import SequenceLabeller
from ..models.anchornet import AnchorNet, AnchorNetConfig
from ..splits.normalizer import Normalizer
from ..splits.windower import SequenceWindower
from .calibration import assess_calibration

_WIN_DUR_S = WINDOW_SIZE_SAMPLES / SAMPLE_RATE_HZ


@torch.no_grad()
def evaluate_split(
    sequences,
    *,
    checkpoint_path: str,
    normalizer_path: str,
    radius_m: float,
    model_cfg: AnchorNetConfig | None = None,
    include_drop: bool = True,
) -> dict:
    net = AnchorNet(model_cfg or AnchorNetConfig())
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net.eval()
    norm = Normalizer.load(normalizer_path)
    win = SequenceWindower(training=False)

    y, mu, logvar, lsig, speeds, usab = [], [], [], [], [], []
    for seq in sequences:
        u = seq.meta.get("usability", "weak")
        if u == "drop" and not include_drop:
            continue
        feats = sequence_model_features(seq, align_sequence_to_vehicle_frame(seq))
        lab = SequenceLabeller(seq, radius_m)
        for w in win.windows(seq):
            window = feats[w.start:w.stop]
            if len(window) < WINDOW_SIZE_SAMPLES:
                continue
            x = norm.transform(window[None]).astype(np.float32)
            out = net(torch.from_numpy(x))
            wl = lab.label(w.start, w.stop)
            y.append(wl.mean_speed_mps)
            mu.append(float(out["velocity_mean_mps"]))
            logvar.append(float(out["velocity_log_variance"]))
            lsig.append(wl.label_sigma_m / _WIN_DUR_S)
            speeds.append(wl.mean_speed_mps)
            usab.append(u)

    y = np.array(y); mu = np.array(mu); logvar = np.array(logvar)
    lsig = np.array(lsig); speeds = np.array(speeds); usab = np.array(usab)
    err = mu - y

    cal = assess_calibration(y, mu, logvar, label_sigma=lsig)

    def _slice(mask):
        if mask.sum() == 0:
            return None
        return {"n": int(mask.sum()),
                "rmse_mps": round(float(np.sqrt(np.mean(err[mask] ** 2))), 4),
                "bias_mps": round(float(np.mean(err[mask])), 4),
                "mean_speed_mps": round(float(np.mean(y[mask])), 3)}

    deciles = np.clip((speeds / (speeds.max() + 1e-6) * 10).astype(int), 0, 9)
    by_decile = {int(d): _slice(deciles == d) for d in range(10)}
    by_usability = {u: _slice(usab == u) for u in ("use", "weak", "drop")}

    return {
        "n_windows": len(y),
        "overall": _slice(np.ones_like(y, bool)),
        "by_speed_decile": by_decile,
        "by_usability": by_usability,
        "calibration": cal.as_dict(),
    }


def evaluate_run(run_dir: str, sequences, *, radius_m: float,
                 normalizer_path: str, out_name: str = "eval_windows.json") -> dict:
    rd = Path(run_dir)
    ckpts = sorted(rd.glob("anchornet_seed*.pt"))
    per_seed = []
    for c in ckpts:
        r = evaluate_split(sequences, checkpoint_path=str(c),
                           normalizer_path=normalizer_path, radius_m=radius_m)
        r["checkpoint"] = c.name
        per_seed.append(r)

    def _ms(path):
        vals = []
        for r in per_seed:
            v = r
            for k in path:
                v = v[k]
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        vals = np.array(vals, dtype=float)
        return {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)}

    summary = {
        "n_seeds": len(per_seed),
        "overall_rmse_mps": _ms(["overall", "rmse_mps"]),
        "overall_bias_mps": _ms(["overall", "bias_mps"]),
        "ece_sigma": _ms(["calibration", "ece_sigma"]),
        "pit_ks": _ms(["calibration", "pit_ks"]),
        "per_seed": per_seed,
    }
    (rd / out_name).write_text(json.dumps(summary, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
    return summary
