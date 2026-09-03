"""Discover synchronised drives on disk and join each phone+vehicle pair into a
SyncedSequence.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .iovnbd import load_smartphone_csv, load_vehicle_csv
from .quality import score_sequence
from .sequence import SyncedSequence, detect_breaks, segments_from_breaks

# group-folder -> driver letter (ml/docs/IO-VNBD-verification.md §1)
_DRIVER_BY_GROUP = {
    "m (driver b)": "B",
    "s (driver a)": "A",
    "y (driver d)": "D",
    "vf (driver e)": "E",
    "vta (driver e)": "E",
    "vtb (driver e)": "E",
    "vw (driver e)": "E",
}
_ROUTE_FAMILY_RE = re.compile(r"^(vta|vtb|vw|vf|s|m|y)", re.IGNORECASE)

SYNC_SUBPATH = Path("Synchronised V abd S datasets") / "Categorised IOVNB Dataset"

# A tail-length mismatch between S and V of more than this fraction of rows gets
# flagged for a manual look; smaller ones (≤1 row is near-universal) are just
# truncated silently. Measured worst case: vfa02 at 0.34%.
LENGTH_MISMATCH_FLAG_FRAC = 0.01


def _iter_pair_dirs(sync_root: Path):
    """Yield (group_folder, sequence_dir) for every dir holding an S-*.csv +
    V-*.csv pair (both > 1 KB, i.e. LFS-materialised)."""
    for group in sorted(p for p in sync_root.iterdir() if p.is_dir()):
        for dirpath, _dirs, files in os.walk(group):
            d = Path(dirpath)
            s = [d / f for f in files if f.lower().startswith("s-") and f.lower().endswith(".csv")]
            v = [d / f for f in files if f.lower().startswith("v-") and f.lower().endswith(".csv")]
            s = [p for p in s if p.stat().st_size > 1024]
            v = [p for p in v if p.stat().st_size > 1024]
            if s and v:
                yield group, d, s[0], v[0]


def _seq_id_from_dir(group: Path, seq_dir: Path) -> str:
    name = seq_dir.name if seq_dir != group else group.name.split(" ")[0]
    name = name.lower()
    name = re.sub(r"^[sv]-", "", name)          # 'V-Vfa01' -> 'vfa01'
    return re.sub(r"[^a-z0-9]", "", name)


def _route_family(seq_id: str) -> str:
    m = _ROUTE_FAMILY_RE.match(seq_id)
    return m.group(1).capitalize() if m else "?"


def load_synced_sequence(group: Path, seq_dir: Path, s_path: Path, v_path: Path) -> SyncedSequence:
    s_res = load_smartphone_csv(s_path)
    v_res = load_vehicle_csv(v_path)
    s_df, v_df = s_res.df, v_res.df

    n = min(len(s_df), len(v_df))
    dropped = abs(len(s_df) - len(v_df))
    s_df = s_df.iloc[:n].reset_index(drop=True)
    v_df = v_df.iloc[:n].reset_index(drop=True)

    df = pd.concat(
        [s_df.add_prefix("phone_"), v_df.add_prefix("veh_")],
        axis=1,
    )
    # synthetic uniform 10 Hz clock — the only timeline downstream code trusts
    df["t_ms"] = np.arange(n, dtype=np.int64) * 100

    breaks = detect_breaks(
        s_df["time_since_start_ms"].to_numpy(),
        v_df["time_of_day_s"].to_numpy(),
    )
    segments = segments_from_breaks(n, breaks)

    seq_id = _seq_id_from_dir(group, seq_dir)
    group_key = group.name.lower()
    warnings = list(s_res.warnings) + list(v_res.warnings)

    quality = score_sequence(df)

    meta = {
        "s_path": str(s_path),
        "v_path": str(v_path),
        "s_rows_raw": s_res.n_rows,
        "v_rows_raw": v_res.n_rows,
        "rows_dropped_to_align": int(dropped),
        "n_breaks": len(breaks),
        "n_segments": len(segments),
        "load_warnings": warnings,
        **quality,
    }
    if dropped > max(1, int(LENGTH_MISMATCH_FLAG_FRAC * n)):
        meta["length_mismatch_flag"] = f"{dropped} rows ({dropped / n:.2%})"

    return SyncedSequence(
        seq_id=seq_id,
        route_family=_route_family(seq_id),
        driver=_DRIVER_BY_GROUP.get(group_key, "?"),
        region="england",
        df=df,
        segments=segments,
        meta=meta,
    )


def discover_sequences(iovnbd_root: str | Path) -> list[SyncedSequence]:
    """Load every materialised synchronised drive under `iovnbd_root`.
    `iovnbd_root` is the IO-VNBD checkout root (contains 'Synchronised V abd S datasets')."""
    sync_root = Path(iovnbd_root) / SYNC_SUBPATH
    if not sync_root.is_dir():
        raise FileNotFoundError(f"synchronised subset not found at {sync_root}")
    seqs: list[SyncedSequence] = []
    seen: set[str] = set()
    for group, seq_dir, s_path, v_path in _iter_pair_dirs(sync_root):
        seq = load_synced_sequence(group, seq_dir, s_path, v_path)
        if seq.seq_id in seen:
            seq.seq_id = f"{seq.seq_id}_{group.name.split(' ')[0].lower()}"
        seen.add(seq.seq_id)
        seqs.append(seq)
    return sorted(seqs, key=lambda s: s.seq_id)
