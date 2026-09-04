"""Synthetic GNSS-outage segment sampling (PRD §6.7 outage protocol).

Outages are synthetic: a held-out sequence has continuous VBOX ground truth
throughout; we simulate losing GNSS for a fixed duration and score the
dead-reckoned trajectory against the truth that was there all along. Durations
30/60/120/180 s match WhONet's published protocol.

The frozen 40-segment golden set (PRD §14.7) is built from this at end of
Week 3 and committed to ml/anchor/golden/. Until then this sampler produces a
deterministic working set from a seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import SAMPLE_RATE_HZ

DURATIONS_S = (30, 60, 120, 180)
_MIN_START_SPEED_MPS = 2.0        # don't start an outage from a dead stop


@dataclass(frozen=True)
class OutageSpec:
    seq_id: str
    seg_index: int
    start_row: int                # first row with "no GNSS"
    duration_s: int
    scenario: str = "unlabelled"  # motorway / roundabout / braking / ... (Week 3)

    @property
    def n_rows(self) -> int:
        return int(self.duration_s * SAMPLE_RATE_HZ)

    @property
    def stop_row(self) -> int:
        return self.start_row + self.n_rows

    def key(self) -> str:
        return f"{self.seq_id}:{self.seg_index}:{self.start_row}:{self.duration_s}"


def sample_outages(sequences, *, seed: int, per_duration_per_seq: int = 2,
                   label_scenarios: bool = True) -> list[OutageSpec]:
    from .scenarios import classify_window

    rng = np.random.default_rng(seed)
    by_id = {s.seq_id: s for s in sequences}
    out: list[OutageSpec] = []
    for seq in sorted(sequences, key=lambda s: s.seq_id):
        speed = seq.df["veh_speed_mps"].to_numpy()
        for si, (a, b) in enumerate(seq.segments):
            for dur in DURATIONS_S:
                need = int(dur * SAMPLE_RATE_HZ)
                lo, hi = a + SAMPLE_RATE_HZ, b - need - SAMPLE_RATE_HZ
                if hi <= lo:
                    continue
                cands = [s for s in range(lo, hi, SAMPLE_RATE_HZ)
                         if speed[s] >= _MIN_START_SPEED_MPS]
                if not cands:
                    continue
                picks = rng.choice(cands, size=min(per_duration_per_seq, len(cands)),
                                   replace=False)
                for p in sorted(int(x) for x in picks):
                    scen = (classify_window(seq, p, p + need)
                            if label_scenarios else "unlabelled")
                    out.append(OutageSpec(seq.seq_id, si, p, dur, scenario=scen))
    return out
