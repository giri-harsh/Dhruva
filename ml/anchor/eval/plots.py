"""Render the eval artefacts to PNG (matplotlib) for slides + the dashboard.

  reliability_diagram(report_id, report_ood)  FR-08 — predicted vs realised σ,
                                              in-distribution and under domain shift
  error_growth(bench_json)                    median/p95 horizontal error vs
                                              outage duration (PRD §6.7: "the single
                                              most informative plot")
  integrity_roc(roc_json)                     detection vs false-rejection per family
  golden_scenarios(baseline_json)             per-scenario drift bars

All write to `ml/eval/plots/`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_OUT = Path(__file__).resolve().parents[3] / "ml" / "eval" / "plots"


def _save(fig, name: str) -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    p = _OUT / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def reliability_diagram(cal_id: dict, cal_ood: dict | None = None,
                        name: str = "reliability.png") -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    lim = 0.0
    for cal, label, style in ((cal_id, "in-distribution (val)", "o-"),
                              (cal_ood, "domain shift (test)", "s--")):
        if not cal:
            continue
        bins = cal.get("bins") or cal.get("calibration", {}).get("bins", [])
        xs = [b["sigma_pred_m"] for b in bins]
        ys = [b["err_rms_m"] for b in bins]
        ax.plot(xs, ys, style, label=f"{label}  (ECE={cal.get('ece_sigma', '?')})")
        lim = max(lim, max(xs + ys, default=0))
    ax.plot([0, lim], [0, lim], "k:", lw=1, label="perfect")
    ax.set_xlabel("predicted σ (m)"); ax.set_ylabel("realised RMS error (m)")
    ax.set_title("Head B calibration (FR-08)")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    return _save(fig, name)


def error_growth(bench: dict, name: str = "error_growth.png") -> Path:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for bid, r in bench.get("baselines", {}).items():
        if r.get("status") != "ok":
            continue
        bd = r["aggregate"]["by_duration"]
        d = sorted(bd, key=int)
        med = [bd[k]["final_error_m"]["median"] for k in d if bd[k].get("final_error_m")]
        ax.plot([int(k) for k in d][:len(med)], med, "o-", label=bid)
    ax.set_xlabel("outage duration (s)"); ax.set_ylabel("median final error (m)")
    ax.set_title("Error growth vs outage duration"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def integrity_roc(roc: dict, name: str = "integrity_roc.png") -> Path:
    fams = {f: r for f, r in roc.get("families", {}).items() if r.get("roc")}
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for f, r in fams.items():
        fr = [p["false_rejection"] for p in r["roc"]]
        de = [p["detection"] for p in r["roc"]]
        ax.plot(fr, de, "o-", ms=3, label=f)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false-rejection rate"); ax.set_ylabel("detection rate")
    ax.set_title(f"GNSS integrity ROC — {roc.get('detector', '')}")
    ax.legend(fontsize=8); ax.set_xlim(0, 0.3); ax.set_ylim(0, 1.02)
    return _save(fig, name)


def golden_scenarios(baseline: dict, name: str = "golden_scenarios.png") -> Path:
    from collections import defaultdict
    agg = defaultdict(list)
    for s in baseline.get("per_segment", []):
        agg[s["scenario"]].append(s["drift_pct"])
    labels = sorted(agg, key=lambda k: -sum(agg[k]) / len(agg[k]))
    med = [sorted(agg[k])[len(agg[k]) // 2] for k in labels]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(labels, med, color="#c0504d")
    ax.axhline(10, color="k", ls="--", lw=1, label="PS 10% bar")
    ax.set_ylabel("median drift %"); ax.set_title("Golden set — drift by scenario")
    ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)
    return _save(fig, name)
