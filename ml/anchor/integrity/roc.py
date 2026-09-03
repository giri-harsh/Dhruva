"""Integrity ROC bench (FR-31).

Runs a GNSS-fault detector over injected attacks and scores:
  * detection rate  = P(flag | attacked fix)
  * false-rejection = P(flag | clean fix)   [on clean, un-attacked sequences]
a ROC curve as the detector threshold sweeps, the operating threshold, and
"the regime that is provably undetected" — the largest attack parameter in each
family for which detection stays at chance.

FR-31 tests Kamal's `ChiSquareGate` — that is the detector we plug in. Until it
exists we use `InnovationResidualDetector` (a NumPy stand-in: flag when the GNSS
fix disagrees with an IMU/last-fix dead-reckoned prediction by more than k*sigma).
The bench is detector-agnostic: `score_detector(detector, ...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..contract import SAMPLE_RATE_HZ
from .attacks import AttackSpec, InjectedTrack, inject, sweep

DT_S = 1.0 / SAMPLE_RATE_HZ


class Detector(Protocol):
    def residuals(self, track: InjectedTrack) -> np.ndarray:
        """Per-sample test statistic; higher = more anomalous. NaN where no fix."""
        ...


@dataclass
class InnovationResidualDetector:
    """Reference stand-in for ChiSquareGate. Predict each fix from the previous
    accepted fix + a constant-velocity model over the gap; the innovation
    magnitude (normalised by an online noise estimate) is the statistic."""
    process_sigma_mps: float = 1.5

    def residuals(self, track: InjectedTrack) -> np.ndarray:
        e, n, valid = track.east_m, track.north_m, track.valid
        N = len(e)
        stat = np.full(N, np.nan)
        last_i = None
        vel = np.zeros(2)
        run_sigma = 3.0
        for i in range(N):
            if not valid[i]:
                continue
            if last_i is not None:
                gap = (i - last_i) * DT_S
                pred = np.array([e[last_i], n[last_i]]) + vel * gap
                innov = np.hypot(e[i] - pred[0], n[i] - pred[1])
                gate_sigma = np.hypot(run_sigma, self.process_sigma_mps * gap)
                stat[i] = innov / max(gate_sigma, 1e-6)
                if stat[i] < 4.0:  # only learn noise / velocity from accepted fixes
                    run_sigma = 0.98 * run_sigma + 0.02 * innov
                    vel = 0.7 * vel + 0.3 * (np.array([e[i], n[i]]) -
                                             np.array([e[last_i], n[last_i]])) / max(gap, DT_S)
            last_i = i
        return stat


def _rates(stat: np.ndarray, attacked: np.ndarray, thr: float):
    flag = stat >= thr
    atk = attacked & np.isfinite(stat)
    cln = (~attacked) & np.isfinite(stat)
    det = float(np.mean(flag[atk])) if atk.any() else np.nan
    fr = float(np.mean(flag[cln])) if cln.any() else np.nan
    return det, fr


@dataclass
class FamilyROC:
    family: str
    swept_param: list[float]
    thresholds: list[float]
    roc: list[dict]                 # per threshold: {thr, detection, false_rejection}
    operating_threshold: float
    detection_at_operating: dict    # param -> detection rate at operating threshold
    provably_undetected_param: float | None


def score_detector(
    detector: Detector,
    clean_seqs,
    attack_seqs,
    *,
    families: dict[str, list[float]],
    seg_len_s: int = 120,
    seed: int = 20260903,
    target_false_rejection: float = 0.02,
) -> dict:
    rng = np.random.default_rng(seed)
    seg = _pick_segment(attack_seqs[0], seg_len_s, rng)

    # false-rejection is measured on CLEAN sequences (no attack)
    clean_stats = []
    for s in clean_seqs:
        cseg = _pick_segment(s, seg_len_s, rng)
        tr = inject(s, AttackSpec("multipath", 0.0), seg=cseg)  # param 0 => no corruption
        clean_stats.append(detector.residuals(tr))
    clean_stat = np.concatenate([c[np.isfinite(c)] for c in clean_stats])
    thresholds = list(np.quantile(clean_stat, np.linspace(0.5, 0.9999, 30)))
    operating = float(np.quantile(clean_stat, 1.0 - target_false_rejection))

    out = {}
    for fam, params in families.items():
        roc_pts, det_at_op, undetected = [], {}, None
        # ROC: aggregate over the mid swept param
        mid = params[len(params) // 2]
        agg_stat, agg_atk = [], []
        for s in attack_seqs:
            aseg = _pick_segment(s, seg_len_s, rng)
            tr = inject(s, AttackSpec(fam, mid, seed=int(rng.integers(1e6))), seg=aseg)
            st = detector.residuals(tr)
            agg_stat.append(st); agg_atk.append(tr.attacked)
        agg_stat = np.concatenate(agg_stat); agg_atk = np.concatenate(agg_atk)
        for thr in thresholds:
            det, fr = _rates(agg_stat, agg_atk, thr)
            roc_pts.append({"thr": round(thr, 3), "detection": _r(det), "false_rejection": _r(fr)})

        for p in params:
            ds = []
            for s in attack_seqs:
                aseg = _pick_segment(s, seg_len_s, rng)
                tr = inject(s, AttackSpec(fam, p, seed=int(rng.integers(1e6))), seg=aseg)
                st = detector.residuals(tr)
                d, _ = _rates(st, tr.attacked, operating)
                if np.isfinite(d):
                    ds.append(d)
            dmean = float(np.mean(ds)) if ds else np.nan
            det_at_op[p] = _r(dmean)
            if (undetected is None) and np.isfinite(dmean) and dmean <= target_false_rejection + 0.03:
                undetected = p

        out[fam] = FamilyROC(
            family=fam, swept_param=list(params), thresholds=[round(t, 3) for t in thresholds],
            roc=roc_pts, operating_threshold=round(operating, 3),
            detection_at_operating=det_at_op, provably_undetected_param=undetected,
        ).__dict__

    return {
        "detector": type(detector).__name__,
        "target_false_rejection": target_false_rejection,
        "operating_threshold": round(operating, 3),
        "seed": seed,
        "families": out,
    }


def _pick_segment(seq, seg_len_s, rng):
    need = seg_len_s * SAMPLE_RATE_HZ
    best = max(seq.segments, key=lambda ab: ab[1] - ab[0])
    a, b = best
    if b - a <= need:
        return (a, b)
    start = int(rng.integers(a, b - need))
    return (start, start + need)


def _r(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)
