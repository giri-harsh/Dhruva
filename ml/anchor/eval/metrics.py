"""Outage trajectory metrics (PRD §6.7).

Implemented now: drift-as-%-of-distance (the PS benchmark, reported first),
final horizontal error, ATE (RMSE after rigid alignment), CTE (max/mean
perpendicular distance to the truth path), and the error-growth series.
RTE and CRSE stubs are marked TODO for the full ablation runner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import path_length, rigid_align_2d


@dataclass
class OutageScore:
    seq_id: str
    duration_s: int
    scenario: str
    distance_travelled_m: float
    final_error_m: float
    drift_pct: float
    ate_m: float
    cte_mean_m: float
    cte_max_m: float
    heading_error_deg_end: float
    error_growth_m: list[float]     # horizontal error sampled each second

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def _pointwise_error(pe, pn, te, tn) -> np.ndarray:
    return np.hypot(pe - te, pn - tn)


def _cross_track(pe, pn, te, tn) -> np.ndarray:
    """Perpendicular distance from each predicted point to the truth polyline."""
    truth = np.c_[te, tn]
    seg_a = truth[:-1]
    seg_b = truth[1:]
    ab = seg_b - seg_a
    ab_len2 = np.sum(ab ** 2, axis=1)
    ab_len2[ab_len2 == 0] = 1e-9
    out = np.empty(len(pe))
    for i, p in enumerate(np.c_[pe, pn]):
        t = np.clip(np.sum((p - seg_a) * ab, axis=1) / ab_len2, 0.0, 1.0)
        proj = seg_a + t[:, None] * ab
        out[i] = np.min(np.hypot(*(p - proj).T))
    return out


def score_outage(
    *,
    seq_id: str,
    duration_s: int,
    scenario: str,
    pred_e: np.ndarray,
    pred_n: np.ndarray,
    truth_e: np.ndarray,
    truth_n: np.ndarray,
    pred_heading_end_rad: float | None = None,
    truth_heading_end_rad: float | None = None,
    sample_rate_hz: int = 10,
) -> OutageScore:
    dist = path_length(truth_e, truth_n)
    err = _pointwise_error(pred_e, pred_n, truth_e, truth_n)
    final_err = float(err[-1])
    drift_pct = float(100.0 * final_err / dist) if dist > 1e-6 else float("nan")

    ae, an = rigid_align_2d(pred_e, pred_n, truth_e, truth_n)
    ate = float(np.sqrt(np.mean(_pointwise_error(ae, an, truth_e, truth_n) ** 2)))

    cte = _cross_track(pred_e, pred_n, truth_e, truth_n)

    if pred_heading_end_rad is None or truth_heading_end_rad is None:
        hdg_err = float("nan")
    else:
        d = np.degrees(pred_heading_end_rad - truth_heading_end_rad)
        hdg_err = float(abs((d + 180.0) % 360.0 - 180.0))

    step = sample_rate_hz
    growth = [float(err[min(i, len(err) - 1)]) for i in range(0, len(err) + 1, step)][1:]

    return OutageScore(
        seq_id=seq_id, duration_s=duration_s, scenario=scenario,
        distance_travelled_m=dist, final_error_m=final_err, drift_pct=drift_pct,
        ate_m=ate, cte_mean_m=float(np.mean(cte)), cte_max_m=float(np.max(cte)),
        heading_error_deg_end=hdg_err, error_growth_m=growth,
    )


def aggregate(scores: list[OutageScore]) -> dict:
    """Group by duration and by scenario, report median + p95 of key metrics."""
    def _stats(sel, field):
        vals = np.array([getattr(s, field) for s in sel if np.isfinite(getattr(s, field))])
        if len(vals) == 0:
            return None
        return {"median": round(float(np.median(vals)), 3),
                "p95": round(float(np.percentile(vals, 95)), 3),
                "n": len(vals)}

    by_dur = {}
    for d in sorted({s.duration_s for s in scores}):
        sel = [s for s in scores if s.duration_s == d]
        by_dur[str(d)] = {f: _stats(sel, f) for f in ("drift_pct", "final_error_m", "ate_m", "cte_max_m")}

    by_scenario = {}
    for sc in sorted({s.scenario for s in scores}):
        sel = [s for s in scores if s.scenario == sc]
        by_scenario[sc] = {f: _stats(sel, f) for f in ("drift_pct", "final_error_m")}

    return {"n_outages": len(scores), "by_duration": by_dur, "by_scenario": by_scenario}
