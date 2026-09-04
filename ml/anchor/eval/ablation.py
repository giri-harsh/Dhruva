"""The 13-row ablation table (PRD §6.7).

Every row is one line on the results slide. Rows 1-8 and 12-13 are ours to run
and score; rows 9-11 need Kamal's filter and are marked external — the runner
leaves a slot for his numbers.

Each row is a training/eval configuration. `ROWS` is the registry; `run_row`
trains it (5 seeds) and evaluates window-level + outage metrics; `build_table`
assembles the committed `ml/eval/ablation_table.json` from the per-row runs.

We deliberately keep this declarative: adding a row is a dict entry, not a code
change, so the table and the PRD stay in step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.anchornet import AnchorNetConfig
from ..train.loop import TrainConfig


@dataclass
class AblationRow:
    n: int
    key: str
    label: str
    owner: str = "ml"                 # "ml" | "kamal" | "cited"
    train: TrainConfig | None = None
    notes: str = ""
    # how the velocity estimate is consumed in the outage bench
    variance_mode: str = "predicted"  # "predicted" | "fixed_R" | "none"


def _cfg(**model_kw) -> TrainConfig:
    tc = TrainConfig()
    tc.model = AnchorNetConfig(**model_kw)
    return tc


ROWS: list[AblationRow] = [
    AblationRow(1, "b2_strapdown", "Strapdown INS only (B2)", owner="ml",
                notes="reference/anchor_ref/strapdown.py — physics-only double "
                      "integration, RUNNABLE. test_id median drift 71/124/117/160% "
                      "@ 30/60/120/180 s (quadratic error growth, PRD §1.2)."),
    AblationRow(2, "b2_nhc", "+ NHC", owner="kamal",
                notes="an NHC-only (no ZUPT) intermediate row -- reference/anchor_ref/"
                      "eskf.py now exists but dispatches NHC/ZUPT together per FR-10/26 "
                      "(NHC suppressed exactly when ZUPT is active), not as a separately "
                      "selectable mode; this row still needs that split, or a decision "
                      "that the table skips straight from B2 to B3"),
    AblationRow(3, "b3_nhc_zupt", "+ NHC + ZUPT (B3)", owner="kamal",
                notes="reference/anchor_ref/eskf.py implemented and runnable "
                      "(android/week3-reference-eskf) -- see ml/tests/test_eskf.py for "
                      "scenario coverage and that file's own README for the ablation "
                      "runner to wire this row's numbers in via ml.anchor.bench.run_baselines"),
    AblationRow(4, "vel_fixed_r", "+ velocity head, fixed R", owner="ml",
                train=_cfg(), variance_mode="fixed_R"),
    AblationRow(5, "vel_pred_var", "+ velocity head, predicted sigma^2 -> R  (primary claim)",
                owner="ml", train=_cfg(), variance_mode="predicted"),
    AblationRow(6, "ctx_head", "+ context head -> adaptive noise", owner="ml",
                train=_cfg(enable_context_head=True), variance_mode="predicted"),
    AblationRow(7, "lambda_c", "lambda_c = 0 vs > 0", owner="ml",
                notes="run with lambda_context=0 and >0, report both"),
    AblationRow(8, "head_d", "+ Head D (learned yaw increment)", owner="ml",
                train=_cfg(enable_yaw_head=True), variance_mode="predicted"),
    AblationRow(9, "map_match", "map matching: forward-only vs fixed-lag Viterbi", owner="kamal"),
    AblationRow(10, "road_manifold", "+ road-manifold constraint", owner="kamal"),
    AblationRow(11, "mag_memory", "+ magnetic route memory", owner="kamal"),
    AblationRow(12, "gru", "GRU variant (rejected alternative)", owner="ml",
                train=_cfg(trunk="gru"), variance_mode="predicted",
                notes="if it wins we say so and switch (PRD §6.5)"),
    AblationRow(13, "ood", "Out-of-distribution (France, Nigeria)", owner="ml",
                notes="row 5 config, evaluated on the unsynchronised France/Nigeria "
                      "phone data with GNSS-derived labels; reported separately"),
]

ROWS_BY_KEY = {r.key: r for r in ROWS}
ML_ROWS = [r for r in ROWS if r.owner == "ml"]


def build_table(row_results: dict[str, dict]) -> dict:
    """row_results: key -> {"window": <evaluate_split summary>, "outage": <agg>}.
    Missing rows (not yet run / Kamal's) are left as placeholders."""
    table = []
    for r in ROWS:
        res = row_results.get(r.key)
        table.append({
            "n": r.n, "key": r.key, "label": r.label, "owner": r.owner,
            "notes": r.notes,
            "status": "done" if res else ("pending" if r.owner == "ml" else "external"),
            "results": res or None,
        })
    return {
        "schema": "PRD §6.7 13-row ablation",
        "rows": table,
        "headline_row": "vel_pred_var",
    }
