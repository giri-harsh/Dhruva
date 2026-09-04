"""Unsynchronised IO-VNBD smartphone sequences — the Stage-1 pre-training corpus
(PRD §6.6 two-stage plan, promoted to primary by the R-02 finding).

Phone-only recordings (no CAN pair). The only speed label available is the
phone's own ~1 Hz GNSS Doppler speed, forward-filled onto the 10 Hz grid — a
weak label with metre-class / ~1-2 m/s speed noise, so windows train with a
large label_sigma. Pre-train on these, then fine-tune on the synchronised
`train` split's clean wheel-speed labels.

--- LEAKAGE DISCIPLINE ---
The pre-train corpus MUST NOT contain any route family or driver that appears in
a held-out split, or the Stage-2 test/val numbers are contaminated. From
ml/anchor/splits/protocol.py:  test_id = Vta, val = Vtb, test_ood_driver =
Driver A (S*), test_repeat_corridor = vfa02/vtb05, excluded = y1 (Driver D).

  PRETRAIN_ELIGIBLE families: Vw, M  (both in the synchronised `train` split)
                              I  (Nigeria), T (France)  — OOD domains
  HELD OUT of pre-train:      Vta, Vtb, Vf, S-*, A-*, Y-*, St-*
  A fixed fraction of the I / T sequences is reserved as `test_ood_region`
  (never pre-trained) so the OOD row is honest.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..contract import SAMPLE_RATE_HZ
from .iovnbd import HeaderMismatchError, load_smartphone_csv
from .sequence import detect_breaks, segments_from_breaks

UNSYNC_S_SUBPATH = Path(
    "Unsynchronised V and S Dataset"
) / "Uncategorised IOVNB (V and S) Dataset" / "S-Dataset"

_PRETRAIN_FAMILIES = {"vw", "m", "i", "t"}
_OOD_FAMILIES = {"i", "t"}
_OOD_HOLDOUT_FRAC = 0.25            # of I and T sequences -> test_ood_region

_REGION = {"i": "nigeria", "t": "france"}
_DRIVER = {"m": "B", "y": "D", "s": "A", "a": "A", "vw": "E", "vta": "E",
           "vtb": "E", "vf": "E", "vfa": "E", "st": "C", "i": "?", "t": "?"}
_FAM_RE = re.compile(r"^(vfa|vta|vtb|vw|st|s3|s|m|y|a|i|t)", re.IGNORECASE)


@dataclass
class UnsyncSequence:
    seq_id: str
    family: str
    region: str
    driver: str
    df: object                      # canonical smartphone DataFrame
    segments: list[tuple[int, int]]
    role: str                       # "pretrain" | "test_ood_region" | "heldout"
    meta: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def duration_s(self) -> float:
        return self.n_rows / SAMPLE_RATE_HZ


def _family(seq_id: str) -> str:
    m = _FAM_RE.match(seq_id)
    if not m:
        return "?"
    f = m.group(1).lower()
    return "s" if f.startswith("s3") else f


def discover_unsync_phone(iovnbd_root: str | Path) -> list[UnsyncSequence]:
    root = Path(iovnbd_root) / UNSYNC_S_SUBPATH
    if not root.is_dir():
        raise FileNotFoundError(f"unsynchronised S-Dataset not found at {root}")
    files = sorted(p for p in root.glob("S-*.csv") if p.stat().st_size > 1024)

    # deterministic OOD holdout: hash the seq_id
    def _is_ood_holdout(seq_id: str) -> bool:
        h = int.from_bytes(bytes(seq_id, "utf8"), "little") % 1000
        return h < int(_OOD_HOLDOUT_FRAC * 1000)

    out: list[UnsyncSequence] = []
    skipped: list[str] = []
    for p in files:
        seq_id = re.sub(r"[^a-z0-9]", "", p.stem.lower().replace("s-", "", 1))
        fam = _family(seq_id)
        try:
            res = load_smartphone_csv(p)
        except HeaderMismatchError as e:
            skipped.append(f"{p.name}: {str(e).splitlines()[0]}")
            continue
        df = res.df
        breaks = detect_breaks(df["time_since_start_ms"].to_numpy(), None)
        segs = segments_from_breaks(len(df), breaks)

        if fam in _OOD_FAMILIES and _is_ood_holdout(seq_id):
            role = "test_ood_region"
        elif fam in _PRETRAIN_FAMILIES:
            role = "pretrain"
        else:
            role = "heldout"

        out.append(UnsyncSequence(
            seq_id=seq_id, family=fam,
            region=_REGION.get(fam, "england"),
            driver=_DRIVER.get(fam, "?"),
            df=df, segments=segs, role=role,
            meta={"path": str(p), "load_warnings": res.warnings,
                  "n_segments": len(segs)},
        ))
    discover_unsync_phone.last_skipped = skipped   # attribute for the report/CLI
    return out


def summarise(seqs: list[UnsyncSequence]) -> dict:
    by_role: dict[str, float] = {}
    by_region: dict[str, float] = {}
    for s in seqs:
        by_role[s.role] = by_role.get(s.role, 0.0) + s.duration_s
        if s.role != "heldout":
            by_region[s.region] = by_region.get(s.region, 0.0) + s.duration_s
    return {
        "n_sequences": len(seqs),
        "hours_by_role": {k: round(v / 3600, 2) for k, v in sorted(by_role.items())},
        "pretrain_hours_by_region": {k: round(v / 3600, 2) for k, v in sorted(by_region.items())},
    }
