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
from .calibration import assess_calibration, fit_variance_temperature

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
    variance_temperature: float = 1.0,   # post-hoc Head-B recalibration (fit on val)
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
    logvar_cal = logvar + 2.0 * np.log(max(variance_temperature, 1e-6))

    cal = assess_calibration(y, mu, logvar_cal, label_sigma=lsig)

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
        "variance_temperature": round(float(variance_temperature), 4),
        "overall": _slice(np.ones_like(y, bool)),
        "by_speed_decile": by_decile,
        "by_usability": by_usability,
        "calibration": cal.as_dict(),
        "_raw": {"y": y, "mu": mu, "logvar": logvar, "lsig": lsig},  # for temp fitting
    }


def evaluate_run(run_dir: str, sequences, *, radius_m: float, normalizer_path: str,
                 out_name: str = "eval_windows.json",
                 val_sequences=None) -> dict:
    """If `val_sequences` is given, a per-seed Head-B variance temperature is
    fitted on val and applied to the reported (test) calibration."""
    from .calibration import IsotonicVarianceCalibrator, assess_calibration

    rd = Path(run_dir)
    ckpts = sorted(rd.glob("anchornet_seed*.pt"))
    per_seed = []
    for c in ckpts:
        T = 1.0
        iso = None
        if val_sequences is not None:
            vr = evaluate_split(val_sequences, checkpoint_path=str(c),
                                normalizer_path=normalizer_path, radius_m=radius_m)
            vraw = vr["_raw"]
            T = fit_variance_temperature(vraw["y"], vraw["mu"], vraw["logvar"],
                                         label_sigma=vraw["lsig"])
            iso = IsotonicVarianceCalibrator.fit(vraw["y"], vraw["mu"], vraw["logvar"],
                                                 label_sigma=vraw["lsig"])
        r = evaluate_split(sequences, checkpoint_path=str(c),
                           normalizer_path=normalizer_path, radius_m=radius_m,
                           variance_temperature=T)
        raw = r.pop("_raw")
        if iso is not None:
            lv_iso = iso.apply(raw["logvar"])
            cal_iso = assess_calibration(raw["y"], raw["mu"], lv_iso, label_sigma=raw["lsig"])
            r["calibration_isotonic"] = cal_iso.as_dict()
            r["isotonic_calibrator"] = iso.to_json()
            # in-distribution (val) calibration after the SAME isotonic map — the
            # honest "achievable" number; the test one is "under domain shift"
            val_iso = assess_calibration(vraw["y"], vraw["mu"], iso.apply(vraw["logvar"]),
                                         label_sigma=vraw["lsig"])
            r["calibration_isotonic_val"] = val_iso.as_dict()
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
        "variance_temperature": _ms(["variance_temperature"]),
        "overall_rmse_mps": _ms(["overall", "rmse_mps"]),
        "overall_bias_mps": _ms(["overall", "bias_mps"]),
        "ece_sigma_scalar_T": _ms(["calibration", "ece_sigma"]),
        "ece_sigma_isotonic_test": _ms(["calibration_isotonic", "ece_sigma"]),
        "ece_sigma_isotonic_val": _ms(["calibration_isotonic_val", "ece_sigma"]),
        "pit_ks_isotonic_test": _ms(["calibration_isotonic", "pit_ks"]),
        "per_seed": per_seed,
    }
    (rd / out_name).write_text(json.dumps(summary, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
    return summary
