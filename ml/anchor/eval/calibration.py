"""Calibration of Head B's predicted variance (FR-08).

FR-08 acceptance: bin predictions by predicted variance; empirical error must
match the predicted distribution with expected calibration error (ECE) below a
target; produce a reliability diagram. "Nobody else will show this" (PRD §6.7) —
so it is measured, plotted, and reported, never asserted.

Two complementary views:
  * variance-bin reliability: bin windows by predicted sigma, compare predicted
    sigma to realised RMS error in each bin. ECE_sigma = mean |pred - realised|
    weighted by bin count, normalised by mean realised error.
  * probabilistic calibration (PIT / quantile): under a correct Gaussian
    (mu, sigma), z = (y - mu) / sigma is N(0,1). Report the fraction of |z| within
    k sigma for k in {0.5, 1, 1.5, 2} against the Gaussian expectation, and the
    KS distance of the PIT histogram from uniform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_NORM_CDF = lambda x: 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x):
    # vectorised Abramowitz-Stegun 7.1.26 (abs err < 1.5e-7)
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


@dataclass
class CalibrationReport:
    n: int
    ece_sigma: float
    ece_sigma_abs_m: float
    pit_ks: float
    coverage: dict          # k -> (empirical, gaussian_expected)
    bins: list              # per-bin: {sigma_pred, err_rms, count, lo, hi}
    overall_rmse: float
    mean_pred_sigma: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "ece_sigma": round(self.ece_sigma, 4),
            "ece_sigma_abs_m": round(self.ece_sigma_abs_m, 4),
            "pit_ks": round(self.pit_ks, 4),
            "coverage": {k: [round(a, 3), round(b, 3)] for k, (a, b) in self.coverage.items()},
            "overall_rmse_m": round(self.overall_rmse, 4),
            "mean_pred_sigma_m": round(self.mean_pred_sigma, 4),
            "bins": self.bins,
        }


def assess_calibration(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_logvar: np.ndarray,
    *,
    label_sigma: np.ndarray | None = None,
    n_bins: int = 10,
) -> CalibrationReport:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.asarray(pred_mean, dtype=np.float64).ravel()
    sig = np.sqrt(np.exp(np.asarray(pred_logvar, dtype=np.float64).ravel()))
    if label_sigma is not None:
        sig = np.sqrt(sig ** 2 + np.asarray(label_sigma, dtype=np.float64).ravel() ** 2)

    err = y - mu
    order = np.argsort(sig)
    edges = np.linspace(0, len(y), n_bins + 1).astype(int)

    bins = []
    num = den = 0.0
    num_abs = 0.0
    for i in range(n_bins):
        idx = order[edges[i]:edges[i + 1]]
        if len(idx) == 0:
            continue
        sp = float(np.mean(sig[idx]))
        er = float(np.sqrt(np.mean(err[idx] ** 2)))
        bins.append({"sigma_pred_m": round(sp, 4), "err_rms_m": round(er, 4),
                     "count": len(idx),
                     "sigma_lo_m": round(float(sig[idx].min()), 4),
                     "sigma_hi_m": round(float(sig[idx].max()), 4)})
        w = len(idx)
        num += w * abs(sp - er) / (er + 1e-6)
        num_abs += w * abs(sp - er)
        den += w

    ece = num / den if den else float("nan")
    ece_abs = num_abs / den if den else float("nan")

    z = err / np.maximum(sig, 1e-9)
    pit = _NORM_CDF(z)
    pit_sorted = np.sort(pit)
    cdf_emp = np.arange(1, len(pit) + 1) / len(pit)
    pit_ks = float(np.max(np.abs(pit_sorted - cdf_emp))) if len(pit) else float("nan")

    coverage = {}
    for k in (0.5, 1.0, 1.5, 2.0):
        emp = float(np.mean(np.abs(z) <= k))
        exp = float(2 * _NORM_CDF(k) - 1.0)
        coverage[str(k)] = (emp, exp)

    return CalibrationReport(
        n=len(y), ece_sigma=ece, ece_sigma_abs_m=ece_abs, pit_ks=pit_ks,
        coverage=coverage, bins=bins,
        overall_rmse=float(np.sqrt(np.mean(err ** 2))),
        mean_pred_sigma=float(np.mean(sig)),
    )


def reliability_diagram_points(report: CalibrationReport):
    """(predicted sigma, realised rms error) per bin — for the dashboard plot.
    A perfectly calibrated model lies on y = x."""
    return [(b["sigma_pred_m"], b["err_rms_m"]) for b in report.bins]


def fit_variance_temperature(
    y_true: np.ndarray, pred_mean: np.ndarray, pred_logvar: np.ndarray,
    *, label_sigma: np.ndarray | None = None, objective: str = "ece",
) -> float:
    """Post-hoc single-scalar Head-B recalibration: pick T > 0 so that
    N(mu, (T*sigma)^2 + label_sigma^2) is calibrated on this (val) set. Apply as
    logvar' = logvar + 2*ln(T) upstream of the manifest — it scales the exported
    variance, NOT a graph change.

    objective="ece"  : 1-D search minimising the variance-bin ECE (what FR-08
                       actually measures — a shape mismatch a scalar can't fully
                       fix will show as a floor here).
    objective="nll"  : closed-form-ish T^2 = mean((y-mu)^2 / var) (+ label floor).
    """
    y = np.asarray(y_true, float).ravel()
    mu = np.asarray(pred_mean, float).ravel()
    lv = np.asarray(pred_logvar, float).ravel()
    ls = None if label_sigma is None else np.asarray(label_sigma, float).ravel()

    if objective == "nll":
        v = np.exp(lv); se = (y - mu) ** 2
        if ls is None:
            return float(np.sqrt(np.mean(se / np.maximum(v, 1e-9))))
        t2 = float(np.mean(np.clip(se - ls ** 2, 0, None) / np.maximum(v, 1e-9)))
        return float(np.sqrt(max(t2, 1e-6)))

    grid = np.geomspace(0.3, 4.0, 60)
    best_t, best = 1.0, np.inf
    for t in grid:
        r = assess_calibration(y, mu, lv + 2 * np.log(t), label_sigma=ls, n_bins=10)
        if r.ece_sigma < best:
            best, best_t = r.ece_sigma, float(t)
    return best_t
