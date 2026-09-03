"""GNSS spoofing / degradation attack injectors (FR-31).

Given a held-out sequence with clean VBOX GNSS, produce a corrupted GNSS track
plus a per-sample ground-truth mask of which fixes are attacked. Four families,
each with one swept parameter:

  step      : an instantaneous position jump of `magnitude_m` at `onset_s`,
              held for the rest of the window. (classic spoof)
  drag      : a slowly growing offset, `rate_mps` m/s, from `onset_s`. (walk-off)
  jam       : GNSS drops out entirely from `onset_s` for `duration_s` (fixes
              marked invalid). swept parameter = duration_s.
  multipath : additive coloured noise, std `sigma_m`, AR(1) rho 0.9, on a
              random `fraction` of fixes. (urban canyon)

The injector does NOT decide detection — it emits (corrupted_track, attack_mask).
`roc.py` runs a detector over it and scores detection vs false-rejection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import SAMPLE_RATE_HZ
from ..eval.geo import lla_to_local_enu

DT_S = 1.0 / SAMPLE_RATE_HZ
_ATTACKS = ("step", "drag", "jam", "multipath")


@dataclass
class AttackSpec:
    family: str
    param: float                 # the swept value (m, m/s, s, or fraction)
    onset_s: float = 20.0
    seed: int = 0

    def __post_init__(self):
        if self.family not in _ATTACKS:
            raise ValueError(f"unknown attack family {self.family!r}")


@dataclass
class InjectedTrack:
    east_m: np.ndarray           # corrupted local-ENU position offered as "GNSS"
    north_m: np.ndarray
    valid: np.ndarray            # bool — is a fix present this sample
    attacked: np.ndarray         # bool ground truth — is this fix corrupted
    truth_east_m: np.ndarray
    truth_north_m: np.ndarray
    spec: AttackSpec


def inject(seq, spec: AttackSpec, *, seg: tuple[int, int] | None = None) -> InjectedTrack:
    a, b = seg or (0, seq.n_rows)
    d = seq.df.iloc[a:b]
    lat = d["veh_gt_lat_deg"].to_numpy()
    lon = d["veh_gt_lon_deg"].to_numpy()
    te, tn = lla_to_local_enu(lat, lon, lat[0], lon[0])
    n = len(te)
    onset = int(spec.onset_s * SAMPLE_RATE_HZ)
    rng = np.random.default_rng(spec.seed)

    ce, cn = te.copy(), tn.copy()
    valid = np.ones(n, dtype=bool)
    attacked = np.zeros(n, dtype=bool)

    if spec.family == "step":
        theta = rng.uniform(0, 2 * np.pi)
        ce[onset:] += spec.param * np.cos(theta)
        cn[onset:] += spec.param * np.sin(theta)
        attacked[onset:] = True
    elif spec.family == "drag":
        theta = rng.uniform(0, 2 * np.pi)
        t = np.arange(n - onset) * DT_S
        ce[onset:] += spec.param * t * np.cos(theta)
        cn[onset:] += spec.param * t * np.sin(theta)
        attacked[onset:] = True
    elif spec.family == "jam":
        dur = int(spec.param * SAMPLE_RATE_HZ)
        valid[onset:onset + dur] = False
        attacked[onset:onset + dur] = True     # "attacked" == "should be flagged unavailable"
    elif spec.family == "multipath":
        # swept param = affected fraction in [0,1]; noise magnitude is fixed
        frac = float(np.clip(spec.param, 0.0, 1.0))
        sigma_m = 8.0
        noise_e = _ar1(n, 0.9, sigma_m, rng)
        noise_n = _ar1(n, 0.9, sigma_m, rng)
        pick = (rng.random(n) < frac) & (np.arange(n) >= onset)
        ce[pick] += noise_e[pick]
        cn[pick] += noise_n[pick]
        attacked[pick] = True

    return InjectedTrack(ce, cn, valid, attacked, te, tn, spec)


def _ar1(n: int, rho: float, sigma: float, rng) -> np.ndarray:
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


def sweep(family: str, values, **kw) -> list[AttackSpec]:
    return [AttackSpec(family=family, param=float(v), **kw) for v in values]
