"""Motion-context labels for Head C (S-15, PRD §6.5).

5-way class set — `idle`, `normal`, `rough`, `impulse`, `handling` — but only
the first THREE have a CAN correlate, so those are the only ones this labeller
emits and the only ones Head C's loss is computed over (masked). `impulse` and
`handling` have no CAN ground truth; Kamal's deterministic on-device detectors
(Hampel-rejection count, remount detector) cover them, and a synthetic-label
variant for Head C is a *separate, clearly-labelled ablation*, never the
reported default.

CAN-derived rules over the 2 s window:
  idle    (0)  mean VBOX speed < 1.0 m/s
  rough   (2)  moving, and the wheel-speed "jitter" (std of the de-trended
               first difference of the mean wheel angular rate) is above
               ROUGH_JITTER_RADPS — a road-roughness excitation proxy that is
               independent of the phone.
  normal  (1)  moving, not rough.

Class ids 3 (impulse) and 4 (handling) are never returned; `context_loss`
computes cross-entropy over logits[:, :3] only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IDLE, NORMAL, ROUGH = 0, 1, 2
CONTEXT_CLASSES = ("idle", "normal", "rough", "impulse", "handling")
N_REAL_CLASSES = 3

IDLE_SPEED_MPS = 1.0
ROUGH_JITTER_RADPS = 0.135      # ~p78 of moving-window jitter on the train split
_WHEELS = ["veh_wheel_fl_radps", "veh_wheel_fr_radps",
           "veh_wheel_rl_radps", "veh_wheel_rr_radps"]


@dataclass
class ContextLabel:
    cls: int                   # 0 / 1 / 2
    jitter: float
    mean_speed_mps: float


class ContextLabeller:
    def __init__(self, seq):
        d = seq.df
        self.wheel_mean = d[_WHEELS].to_numpy().mean(axis=1)
        self.speed = d["veh_speed_mps"].to_numpy()

    def label(self, start: int, stop: int) -> ContextLabel:
        spd = float(np.nanmean(self.speed[start:stop]))
        wm = self.wheel_mean[start:stop]
        dwm = np.diff(wm)
        jitter = float(np.std(dwm - np.nanmean(dwm))) if len(dwm) else 0.0
        if spd < IDLE_SPEED_MPS:
            cls = IDLE
        elif jitter > ROUGH_JITTER_RADPS:
            cls = ROUGH
        else:
            cls = NORMAL
        return ContextLabel(cls=cls, jitter=round(jitter, 4), mean_speed_mps=round(spd, 3))
