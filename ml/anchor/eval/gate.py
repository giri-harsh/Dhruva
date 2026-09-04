"""Week-5 gate evaluation for a trained run (PRD §8 / §20.1).

    python -m anchor.eval.gate --run ml/train/runs/<name> [--split test_id]

Produces, with full provenance:
  <run>/eval_windows.json        window-level speed RMSE + calibration, mean±std over seeds
  <run>/gate.json                the gate summary (ANCHOR-Net DR vs B1, calibration verdict)
  ml/eval/calibration_report.json  reliability-diagram points for the dashboard
  ml/bench/results/baselines_<split>_anchornet.json

Gate criteria the PRD sets (§20.1 Week-5): a >=20% relative drift improvement
over B3 with acceptable calibration. B3 is Kamal's filter — until it is wired in
we report vs B1 (the honest zero-line) and flag that B3 is the real bar.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..data.labels import fit_wheel_radius
from ..data.sync import discover_sequences
from ..splits.protocol import assign_all
from .evaluate import evaluate_run

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IOVNBD = os.environ.get("IOVNBD_ROOT", str(_ROOT / "data" / "raw" / "IO-VNBD"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--iovnbd-root", default=_DEFAULT_IOVNBD)
    args = ap.parse_args()

    run_dir = Path(args.run) if Path(args.run).is_absolute() else _ROOT / args.run
    norm_path = _ROOT / "ml" / "splits" / "normalizer_train.json"

    seqs = discover_sequences(args.iovnbd_root)
    splits = assign_all(seqs)
    radius = fit_wheel_radius(splits["train"]).radius_m

    # 1. window-level + calibration, over all seeds. Head-B variance temperature
    #    is fitted on `val` and applied to the reported (test) calibration.
    win = evaluate_run(str(run_dir), splits[args.split], radius_m=radius,
                       normalizer_path=str(norm_path), val_sequences=splits["val"])
    (_ROOT / "ml" / "eval").mkdir(parents=True, exist_ok=True)
    best = min(win["per_seed"], key=lambda r: r["overall"]["rmse_mps"])
    cal_report = {
        "provenance": {"run": run_dir.name, "checkpoint": best["checkpoint"],
                       "split": args.split,
                       "generated_utc": datetime.now(timezone.utc).isoformat()},
        "scalar_T": {"temperature": best.get("variance_temperature"),
                     **best["calibration"]},
        "isotonic_test_ood": best.get("calibration_isotonic"),
        "isotonic_val_indist": best.get("calibration_isotonic_val"),
        "calibrator": best.get("isotonic_calibrator"),
    }
    (_ROOT / "ml" / "eval" / "calibration_report.json").write_text(
        json.dumps(cal_report, indent=2) + "\n", encoding="utf-8", newline="\n")

    # 2. outage bench: ANCHOR-Net DR vs B1 (same harness)
    from ..bench.run_baselines import run as run_bench
    bench_out = _ROOT / "ml" / "bench" / "results" / f"baselines_{args.split}_anchornet.json"
    bench = run_bench(args.split, args.seed, args.iovnbd_root, bench_out, anchornet_run=str(run_dir))

    b1 = bench["baselines"].get("B1", {}).get("aggregate", {}).get("by_duration", {})
    an = bench["baselines"].get("ANCHORNET", {}).get("aggregate", {}).get("by_duration", {})
    rows = []
    for d in sorted(b1, key=int):
        b1d = b1[d]["drift_pct"]["median"] if b1[d].get("drift_pct") else None
        and_ = an.get(d, {}).get("drift_pct", {}).get("median") if an.get(d) else None
        rel = (round(100 * (b1d - and_) / b1d, 1) if (b1d and and_) else None)
        rows.append({"duration_s": int(d), "B1_drift_pct": b1d,
                     "ANCHORNET_drift_pct": and_, "rel_improvement_vs_B1_pct": rel})

    gate = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": run_dir.name,
        "split": args.split,
        "window_level": {
            "overall_rmse_mps": win["overall_rmse_mps"],
            "overall_bias_mps": win["overall_bias_mps"],
            "ece_sigma_scalar_T": win["ece_sigma_scalar_T"],
            "ece_sigma_isotonic_val_indist": win["ece_sigma_isotonic_val"],
            "ece_sigma_isotonic_test_ood": win["ece_sigma_isotonic_test"],
            "variance_temperature": win["variance_temperature"],
        },
        "drift_vs_B1": rows,
        "fr24_rejected_windows": bench["baselines"].get("ANCHORNET", {}).get("fr24_rejected_windows"),
        "note": "B3 (Kamal's ESKF+NHC+ZUPT) is the real Week-5 bar; B1 shown here "
                "as the honest zero-line until B3 is wired into this harness.",
    }
    (run_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n",
                                       encoding="utf-8", newline="\n")
    (_ROOT / "ml" / "eval" / "gate_summary.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8", newline="\n")

    # render the plots for slides + the dashboard
    try:
        from . import plots
        plots.reliability_diagram(cal_report["isotonic_val_indist"],
                                  cal_report["isotonic_test_ood"])
        plots.error_growth(bench)
        gj = json.loads((_ROOT / "ml" / "golden" / "public_baseline.json").read_text())
        plots.golden_scenarios(gj)
        rj = _ROOT / "ml" / "eval" / "integrity_roc.json"
        if rj.is_file():
            plots.integrity_roc(json.loads(rj.read_text()))
        print("wrote ml/eval/plots/*.png")
    except Exception as e:  # plotting must never fail the gate
        print(f"(plots skipped: {e})")

    print(json.dumps(gate, indent=2))
    print(f"\nwrote {run_dir}/gate.json, ml/eval/gate_summary.json, "
          f"ml/eval/calibration_report.json")


if __name__ == "__main__":
    main()
