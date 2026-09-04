"""Assemble ml/eval/ablation_table.json (PRD §6.7's 13-row table) from whatever
row results exist so far. Kamal's rows (2, 3, 9-11) stay 'external'; unrun
ml rows stay 'pending'.

    python -m anchor.eval.ablation_run [--iovnbd-root PATH]

Reads:
  ml/train/runs/week3_twostage/{summary,gate}.json  -> row 5 (primary claim)
  ml/train/runs/ctx_lc02/summary.json               -> row 6
  ml/train/runs/{ctx_lc02,ctx_lc00}/summary.json    -> row 7 (lambda_c)
  ml/train/runs/gru/summary.json                    -> row 12
  ml/bench/results/baselines_test_id_anchornet.json -> rows 1 (B2), 4 vs 5 drift
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .ablation import ROWS, build_table

_ROOT = Path(__file__).resolve().parents[3]
_RUNS = _ROOT / "ml" / "train" / "runs"
_OUT = _ROOT / "ml" / "eval" / "ablation_table.json"


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.is_file() else None


def _summary(name: str) -> dict | None:
    s = _load(_RUNS / name / "summary.json")
    if not s:
        return None
    sm = s["summary"]
    return {"val_rmse_mps": sm.get("val_rmse_mps"),
            "val_ctx_acc": sm.get("val_ctx_acc"),
            "lambda_context": sm.get("lambda_context"),
            "n_params": sm.get("n_params")}


def collect() -> dict[str, dict]:
    out: dict[str, dict] = {}
    twostage_gate = _load(_RUNS / "week3_twostage" / "gate.json")
    ts = _summary("week3_twostage")
    if ts:
        out["vel_pred_var"] = {"window": ts,
                               "outage": (twostage_gate or {}).get("drift_vs_B1")}
        out["vel_fixed_r"] = {"note": "same model as row 5; fixed-R vs predicted-sigma^2 "
                                      "is a fusion-side choice — realised inside Kamal's "
                                      "filter (needs B3). Standalone DR uses predicted var."}
    ctx = _summary("ctx_lc02")
    if ctx:
        out["ctx_head"] = {"window": ctx}
    lc0, lc02 = _summary("ctx_lc00"), _summary("ctx_lc02")
    if lc0 and lc02:
        out["lambda_c"] = {"lambda_c=0": lc0["val_rmse_mps"],
                           "lambda_c=0.2": lc02["val_rmse_mps"],
                           "ctx_acc@lambda_c=0.2": lc02["val_ctx_acc"],
                           "verdict": _lambda_verdict(lc0, lc02)}
    gru = _summary("gru")
    if gru:
        out["gru"] = {"window": gru}
    bench = _load(_ROOT / "ml" / "bench" / "results" / "baselines_test_id_anchornet.json")
    if bench:
        b2 = bench["baselines"].get("B2", {}).get("aggregate", {}).get("by_duration")
        if b2:
            out["b2_strapdown"] = {"outage_by_duration": {
                d: b2[d]["drift_pct"]["median"] for d in b2 if b2[d].get("drift_pct")}}
    return out


_CTX_MAJORITY_ACC = 0.68        # val "always predict normal" rate (see context_labels)


def _lambda_verdict(lc0: dict, lc02: dict) -> str:
    a = (lc0["val_rmse_mps"] or {}).get("mean")
    b = (lc02["val_rmse_mps"] or {}).get("mean")
    acc = (lc02["val_ctx_acc"] or {}).get("mean", 0.0)
    if a is None or b is None:
        return "pending"
    hurts = b > a + 0.03
    ctx_works = acc > _CTX_MAJORITY_ACC + 0.05
    if not hurts and ctx_works:
        return (f"keep — lambda_c=0.2 doesn't hurt velocity ({b:.3f} vs {a:.3f}) "
                f"and Head C ({acc:.2f}) beats majority-class ({_CTX_MAJORITY_ACC})")
    return (f"CUT Head C — velocity {b:.3f} vs {a:.3f} m/s "
            f"({'worse' if hurts else 'noise'}), and Head C acc {acc:.2f} "
            f"~ majority-class {_CTX_MAJORITY_ACC} (CAN roughness label not "
            f"discriminative on IO-VNBD). Fall back to fixed R + Kamal's "
            f"deterministic detectors (PRD §6.5).")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    table = build_table(collect())
    table["generated_utc"] = datetime.now(timezone.utc).isoformat()
    _OUT.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8", newline="\n")
    done = sum(1 for r in table["rows"] if r["status"] == "done")
    print(f"wrote {_OUT}  ({done}/{len(ROWS)} rows filled)")
    for r in table["rows"]:
        print(f"  {r['n']:2d} {r['key']:20s} [{r['status']}]")


if __name__ == "__main__":
    main()
