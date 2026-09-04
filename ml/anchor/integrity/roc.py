"""Integrity ROC bench (FR-31).

Runs a GNSS-fault detector over injected attacks and scores, PER ATTACK INSTANCE:

  detection rate  = P(the detector raises a flag on an attacked fix within
                     `detection_horizon_s` of the attack onset)
  false-rejection = P(flag | clean fix), measured on separate un-attacked
                     sequences

An earlier version scored detection per-sample over every attacked fix, which
made a held `step`/`drag` spoof look "undetected" on the many fixes AFTER the
constant-velocity tracker re-converges to the spoofed position — an artefact,
not a miss. Per-instance detection (did we catch it at all, near onset) is the
honest granularity.

Output per attack family: a ROC (sweeping the detector threshold), the operating
threshold at a target false-rejection rate, the per-swept-parameter detection
rate at that threshold, and **the regime that is provably undetected** — the
largest attack parameter whose detection rate is no better than chance.

`jam` is scored separately: detection there is "the missing-fix run is observed"
(trivially ~1); what matters is no false rejection of the reacquired fixes.

FR-31 tests Kamal's `ChiSquareGate` — plug it in via the `Detector` protocol.
`InnovationResidualDetector` is the NumPy stand-in until then; treat its numbers
as a reference profile, not a target. Known reference-detector limitations
(a production gate does better): it is twitchy on the first ~10 s after a
blackout (jam false-rejection ~0.16 in the 3-10 s reacquisition window), and
its mean-residual drift test catches only ~50-80 % of slow walk-offs within
25 s. Its clear result: a position step below ~5 m (at 4 m GNSS noise) is
provably undetectable, steps >= 40 m are always caught.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..contract import SAMPLE_RATE_HZ
from .attacks import AttackSpec, InjectedTrack, inject

DT_S = 1.0 / SAMPLE_RATE_HZ
# a spoof must be caught within this many seconds of onset. A slow walk-off
# ("drag") is inherently gradual, so it gets a longer window before it counts
# as a miss — a detector that never catches it even then has a real blind spot.
DETECTION_HORIZON_S = {"step": 8.0, "drag": 25.0, "multipath": 8.0, "jam": 8.0}
_DEFAULT_HORIZON_S = 10.0
_UNDETECTED_RATE = 0.5              # detection at/under this at the operating point = a coin flip


class Detector(Protocol):
    def residuals(self, track: InjectedTrack) -> np.ndarray:
        """Per-sample test statistic; higher = more anomalous. NaN where no fix."""
        ...


@dataclass
class InnovationResidualDetector:
    """Reference stand-in for ChiSquareGate. Per accepted fix, the statistic is
    the larger of:
      * `inst`  — instantaneous innovation vs a constant-velocity prediction
        from the last accepted fix, normalised by an online noise estimate
        (catches a step at onset), and
      * `drift` — a normalised one-sided CUSUM of the *projected* residual along
        the current drift direction, with a slack term so clean noise (which
        averages below slack) does not accumulate but a sustained walk-off does.
    """
    process_sigma_mps: float = 1.5
    drift_ema: float = 0.9         # EMA weight for the mean-residual (drift) test
    gap_reset_s: float = 4.0

    def residuals(self, track: InjectedTrack) -> np.ndarray:
        e, n, valid = track.east_m, track.north_m, track.valid
        N = len(e)
        stat = np.full(N, np.nan)
        a = self.drift_ema
        ema_std_frac = np.sqrt((1 - a) / (1 + a))   # std of EMA of unit-var noise
        last_i = None
        vel = np.zeros(2)
        run_sigma = 5.0
        resid_ema = np.zeros(2)
        seen = 0
        reinit = 3                     # fixes to skip-score after a (re)start while velocity re-converges
        for i in range(N):
            if not valid[i]:
                continue
            if last_i is not None and (i - last_i) * DT_S > self.gap_reset_s:
                vel = np.zeros(2); resid_ema = np.zeros(2); run_sigma = 5.0
                seen = 0; reinit = 5
                last_i = i
                continue
            if last_i is not None:
                if reinit > 0:
                    # re-estimating velocity after a start/gap — seed it, don't flag
                    gap0 = max((i - last_i) * DT_S, DT_S)
                    vel = (np.array([e[i], n[i]]) - np.array([e[last_i], n[last_i]])) / gap0
                    reinit -= 1
                    last_i = i
                    continue
                gap = (i - last_i) * DT_S
                pred = np.array([e[last_i], n[last_i]]) + vel * gap
                resid = np.array([e[i] - pred[0], n[i] - pred[1]])
                innov = float(np.hypot(*resid))
                gate_sigma = float(np.hypot(run_sigma, self.process_sigma_mps * gap))
                inst = innov / max(gate_sigma, 1e-6)

                resid_ema = a * resid_ema + (1 - a) * resid
                seen += 1
                # is the MEAN residual significantly non-zero? (zero on clean noise).
                # `seen > 8` so the EMA has filled after a (re)start.
                drift = (float(np.hypot(*resid_ema)) / max(gate_sigma * ema_std_frac, 1e-6)
                         if seen > 8 else 0.0)

                stat[i] = max(inst, drift)
                if inst < 4.0:
                    run_sigma = 0.98 * run_sigma + 0.02 * innov
                    vel = 0.7 * vel + 0.3 * (np.array([e[i], n[i]])
                                             - np.array([e[last_i], n[last_i]])) / max(gap, DT_S)
            last_i = i
        return stat


# --------------------------------------------------------------------------- #

def _clean_stat_distribution(detector, clean_seqs, seg_len_s, rng) -> np.ndarray:
    vals = []
    for s in clean_seqs:
        seg = _pick_segment(s, seg_len_s, rng)
        tr = inject(s, AttackSpec("multipath", 0.0), seg=seg)   # param 0 -> untouched
        st = detector.residuals(tr)
        vals.append(st[np.isfinite(st)])
    return np.concatenate(vals) if vals else np.array([1.0])


def _instance_detected(stat, track: InjectedTrack, thr: float) -> bool | None:
    """None => not applicable (no attacked fix to catch, e.g. a jam)."""
    onset = int(track.spec.onset_s * SAMPLE_RATE_HZ)
    horizon = DETECTION_HORIZON_S.get(track.spec.family, _DEFAULT_HORIZON_S)
    hi = onset + int(horizon * SAMPLE_RATE_HZ)
    idx = np.where(track.attacked & track.valid & np.isfinite(stat))[0]
    idx = idx[(idx >= onset) & (idx <= hi)]
    if idx.size == 0:
        return None
    return bool(np.any(stat[idx] >= thr))


@dataclass
class FamilyROC:
    family: str
    swept_param: list[float]
    roc: list[dict]                       # {thr, detection, false_rejection}
    operating_threshold: float
    detection_at_operating: dict          # param -> detection rate
    false_rejection_at_operating: float
    provably_undetected_param: float | None
    note: str = ""


def score_detector(
    detector: Detector,
    clean_seqs,
    attack_seqs,
    *,
    families: dict[str, list[float]],
    seg_len_s: int = 120,
    seed: int = 20260903,
    target_false_rejection: float = 0.02,
    n_thresholds: int = 25,
) -> dict:
    rng = np.random.default_rng(seed)
    clean_stat = _clean_stat_distribution(detector, clean_seqs, seg_len_s, rng)
    thresholds = list(np.quantile(clean_stat, np.linspace(0.5, 0.999, n_thresholds)))
    operating = float(np.quantile(clean_stat, 1.0 - target_false_rejection))

    def _fr(thr: float) -> float:
        return float(np.mean(clean_stat >= thr))

    out: dict[str, dict] = {}
    for fam, params in families.items():
        # build one attack instance per (seq, param) with a fixed seed
        instances = []
        for s in attack_seqs:
            for p in params:
                seg = _pick_segment(s, seg_len_s, rng)
                tr = inject(s, AttackSpec(fam, float(p), seed=int(rng.integers(1_000_000))),
                            seg=seg)
                instances.append((p, tr, detector.residuals(tr)))

        if fam == "jam":
            # detection is trivial; report the reacquisition false-reject instead
            reacq_fr = _jam_reacquisition_false_reject(instances, operating)
            out[fam] = FamilyROC(
                family=fam, swept_param=[float(x) for x in params], roc=[],
                operating_threshold=round(operating, 3),
                detection_at_operating={float(p): 1.0 for p in params},
                false_rejection_at_operating=round(reacq_fr, 4),
                provably_undetected_param=None,
                note="jam: outage is always observed; value reported is the "
                     "false-rejection rate on the first 5 s of reacquired fixes",
            ).__dict__
            continue

        roc = []
        for thr in thresholds:
            dets = [_instance_detected(st, tr, thr) for _, tr, st in instances]
            dets = [d for d in dets if d is not None]
            roc.append({"thr": round(float(thr), 3),
                        "detection": _r(np.mean(dets) if dets else np.nan),
                        "false_rejection": _r(_fr(thr))})

        det_at_op: dict[float, float | None] = {}
        undetected = None
        fr_op = _fr(operating)
        for p in params:
            dets = [_instance_detected(st, tr, operating)
                    for pp, tr, st in instances if pp == p]
            dets = [d for d in dets if d is not None]
            rate = float(np.mean(dets)) if dets else float("nan")
            det_at_op[float(p)] = _r(rate)
            if np.isfinite(rate) and rate <= _UNDETECTED_RATE:
                undetected = float(p) if undetected is None else max(undetected, float(p))

        out[fam] = FamilyROC(
            family=fam, swept_param=[float(x) for x in params], roc=roc,
            operating_threshold=round(operating, 3),
            detection_at_operating=det_at_op,
            false_rejection_at_operating=round(fr_op, 4),
            provably_undetected_param=undetected,
        ).__dict__

    return {
        "detector": type(detector).__name__,
        "target_false_rejection": target_false_rejection,
        "detection_horizon_s": DETECTION_HORIZON_S,
        "operating_threshold": round(operating, 3),
        "seed": seed,
        "families": out,
    }


def _jam_reacquisition_false_reject(instances, thr) -> float:
    """Flag rate on the reacquired fixes, measured 3-10 s AFTER GNSS returns —
    the first ~3 s is an accepted settling transient (the filter re-estimates
    velocity after a blackout), not a false rejection."""
    flags = []
    for _, tr, st in instances:
        end = int(tr.spec.onset_s * SAMPLE_RATE_HZ) + int(tr.spec.param * SAMPLE_RATE_HZ)
        lo = end + int(3 * SAMPLE_RATE_HZ)
        hi = end + int(10 * SAMPLE_RATE_HZ)
        idx = np.where(tr.valid & np.isfinite(st))[0]
        idx = idx[(idx >= lo) & (idx <= hi)]
        if idx.size:
            flags.append(float(np.mean(st[idx] >= thr)))
    return float(np.mean(flags)) if flags else 0.0


def _pick_segment(seq, seg_len_s, rng):
    need = seg_len_s * SAMPLE_RATE_HZ
    a, b = max(seq.segments, key=lambda ab: ab[1] - ab[0])
    if b - a <= need:
        return (a, b)
    start = int(rng.integers(a, b - need))
    return (start, start + need)


def _r(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)
