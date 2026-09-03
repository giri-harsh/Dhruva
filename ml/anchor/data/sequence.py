"""SyncedSequence: one synchronised drive (phone + vehicle), row-aligned, with
internal time-break segmentation.

The synchronised subset is row-indexed at a nominal 10 Hz — the dataset authors
already paired phone and CAN rows. We keep that pairing (do NOT try to resample
on the phone's `time_since_start_ms`, which has clock resets and duplicate
stamps). What we DO add:

  * truncate a phone/vehicle pair to the shorter length (small tail mismatch on
    ~7 of 70 pairs, worst 0.34 %) and record the drop,
  * segment the sequence at large timestamp discontinuities in EITHER stream,
    so a training/eval window never straddles a recording gap or clock reset
    (same rule the split protocol applies at sequence boundaries).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contract import SAMPLE_RATE_HZ

NOMINAL_DT_MS = 1000.0 / SAMPLE_RATE_HZ            # 100 ms
BREAK_GAP_MS = 5 * NOMINAL_DT_MS                   # >500 ms gap => segment boundary


@dataclass
class SyncedSequence:
    seq_id: str                 # e.g. "s3b", "vta29", "m", "vw14a"
    route_family: str           # "S", "M", "Y", "Vf", "Vta", "Vtb", "Vw"
    driver: str                 # "A" | "B" | "D" | "E"
    region: str                 # "england" for the whole synchronised subset
    df: pd.DataFrame            # phone_* and veh_* columns, one row per 10 Hz sample
    segments: list[tuple[int, int]]   # [start, end) row ranges with no internal break
    meta: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def duration_s(self) -> float:
        return self.n_rows / SAMPLE_RATE_HZ

    def segment_frames(self):
        for a, b in self.segments:
            yield a, b, self.df.iloc[a:b]


BACKWARD_JUMP_MS = -200.0   # small negative dt = stamp jitter; large = clock reset


def detect_breaks(phone_t_ms: np.ndarray, veh_t_s: np.ndarray) -> list[int]:
    """Row indices at which a new segment starts (the break is between index-1
    and index). Union of *real* discontinuities in either clock.

    Duplicate timestamps (dt == 0) and tiny negative jitter are NOT breaks —
    both phone clocks in this dataset stutter at 10 Hz without any true gap
    (vtb01's phone stamp repeats constantly). Only a backward jump past
    BACKWARD_JUMP_MS (clock reset, e.g. S-M) or a forward gap past
    BREAK_GAP_MS (recording paused) starts a new segment.
    """
    breaks: set[int] = set()
    for t, scale in ((phone_t_ms, 1.0), (veh_t_s, 1000.0)):
        if t is None:
            continue
        d = np.diff(np.asarray(t, dtype=np.float64)) * scale
        bad = np.where(~np.isfinite(d) | (d < BACKWARD_JUMP_MS) | (d > BREAK_GAP_MS))[0]
        for i in bad:
            breaks.add(int(i) + 1)
    return sorted(breaks)


def segments_from_breaks(n_rows: int, breaks: list[int]) -> list[tuple[int, int]]:
    bounds = [0] + [b for b in breaks if 0 < b < n_rows] + [n_rows]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]
