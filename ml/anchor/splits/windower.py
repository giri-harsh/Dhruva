"""SequenceWindower — the leakage-safe windowing rule as code.

PRD-ML-BACKEND.md §6.2 / §6.4, satisfied literally:

  * a window never crosses a sequence boundary  (we split whole sequences, so
    this is automatic — but also never crosses an INTERNAL segment boundary,
    i.e. a clock reset / recording gap detected by the loader),
  * a 10-second guard band is dropped at every boundary (start and end of every
    segment) — sensor settling, alignment transients, and any residual
    sync error at a discontinuity are excluded,
  * training stride is 0.5 s (75 % overlap) — legitimate WITHIN a split;
    inference stride is one whole window (no overlap).

`test_no_cross_sequence_windows` in ml/tests/ asserts the boundary rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import SAMPLE_RATE_HZ, WINDOW_SIZE_SAMPLES

GUARD_BAND_S = 10.0
GUARD_BAND_SAMPLES = int(round(GUARD_BAND_S * SAMPLE_RATE_HZ))   # 100
TRAIN_STRIDE_SAMPLES = int(round(0.5 * SAMPLE_RATE_HZ))          # 5  (0.5 s)
INFER_STRIDE_SAMPLES = WINDOW_SIZE_SAMPLES                       # 20 (no overlap)


@dataclass(frozen=True)
class Window:
    seq_id: str
    seg_index: int
    start: int          # row index into SyncedSequence.df, inclusive
    stop: int           # exclusive; stop - start == WINDOW_SIZE_SAMPLES

    @property
    def center_row(self) -> int:
        return (self.start + self.stop) // 2


def iter_windows_for_segment(
    seq_id: str,
    seg_index: int,
    seg_start: int,
    seg_stop: int,
    *,
    stride: int,
    guard: int = GUARD_BAND_SAMPLES,
    window: int = WINDOW_SIZE_SAMPLES,
):
    """Yield Windows fully inside [seg_start + guard, seg_stop - guard)."""
    lo = seg_start + guard
    hi = seg_stop - guard
    if hi - lo < window:
        return
    for start in range(lo, hi - window + 1, stride):
        yield Window(seq_id=seq_id, seg_index=seg_index, start=start, stop=start + window)


class SequenceWindower:
    def __init__(self, *, training: bool):
        self.training = training
        self.stride = TRAIN_STRIDE_SAMPLES if training else INFER_STRIDE_SAMPLES

    def windows(self, seq) -> list[Window]:
        out: list[Window] = []
        for i, (a, b) in enumerate(seq.segments):
            out.extend(iter_windows_for_segment(
                seq.seq_id, i, a, b, stride=self.stride))
        return out

    def count(self, seq) -> int:
        return sum(1 for _ in self.windows(seq))
