import numpy as np
import pytest

from anchor.contract import WINDOW_SIZE_SAMPLES
from anchor.splits.windower import (
    GUARD_BAND_SAMPLES,
    SequenceWindower,
    iter_windows_for_segment,
)


def test_windows_never_cross_a_segment_boundary():
    # two segments [0,1000) and [1000,2500)
    segs = [(0, 1000), (1000, 2500)]
    wins = []
    for i, (a, b) in enumerate(segs):
        wins += list(iter_windows_for_segment("seq", i, a, b, stride=5))
    assert wins, "expected some windows"
    for w in wins:
        # entirely within exactly one segment, and inside its guard band
        seg = segs[w.seg_index]
        assert seg[0] + GUARD_BAND_SAMPLES <= w.start
        assert w.stop <= seg[1] - GUARD_BAND_SAMPLES
        assert w.stop - w.start == WINDOW_SIZE_SAMPLES


def test_short_segment_yields_nothing():
    # segment shorter than 2*guard + window produces no windows
    n = 2 * GUARD_BAND_SAMPLES + WINDOW_SIZE_SAMPLES - 1
    assert list(iter_windows_for_segment("s", 0, 0, n, stride=5)) == []


def test_train_stride_overlaps_infer_stride_does_not():
    tr = SequenceWindower(training=True)
    inf = SequenceWindower(training=False)
    assert tr.stride == 5           # 0.5 s
    assert inf.stride == WINDOW_SIZE_SAMPLES  # 2.0 s, no overlap


@pytest.mark.usefixtures("sequences")
def test_no_cross_sequence_windows(sequences):
    """FR / PRD §6.2 hygiene rule 1: no window crosses a sequence or internal
    segment boundary, on the REAL data."""
    win = SequenceWindower(training=True)
    for seq in sequences:
        seg_bounds = seq.segments
        for w in win.windows(seq):
            a, b = seg_bounds[w.seg_index]
            assert a <= w.start < w.stop <= b, (
                f"{seq.seq_id}: window [{w.start},{w.stop}) escapes segment [{a},{b})"
            )
            assert w.start >= a + GUARD_BAND_SAMPLES
            assert w.stop <= b - GUARD_BAND_SAMPLES
        # windows are within the sequence
        for w in win.windows(seq):
            assert 0 <= w.start and w.stop <= seq.n_rows
