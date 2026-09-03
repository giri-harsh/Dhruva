"""Build the committed sequence index for the IO-VNBD synchronised subset.

Output: ml/anchor/data/sequence_index.json — one record per synchronised drive
with driver / route family / duration / segment count / quality scores / source
file SHA-256s. This is a COMMITTED provenance artefact: every split manifest and
every reported number traces back to an exact, immutable list of sequences.

Usage:
    python -m anchor.data.build_index [--iovnbd-root PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .sync import discover_sequences

_DEFAULT_ROOT = os.environ.get("IOVNBD_ROOT", "data/raw/IO-VNBD")
_DEFAULT_OUT = Path(__file__).parent / "sequence_index.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(iovnbd_root: str | Path, out_path: str | Path) -> dict:
    seqs = discover_sequences(iovnbd_root)
    records = []
    for s in seqs:
        m = s.meta
        records.append({
            "seq_id": s.seq_id,
            "route_family": s.route_family,
            "driver": s.driver,
            "region": s.region,
            "n_rows": s.n_rows,
            "duration_s": round(s.duration_s, 1),
            "n_segments": m["n_segments"],
            "segments": s.segments,
            "rows_dropped_to_align": m["rows_dropped_to_align"],
            "length_mismatch_flag": m.get("length_mismatch_flag"),
            "vib_speed_corr": m["vib_speed_corr"],
            "lsq_yaw_r2": m["lsq_yaw_r2"],
            "sync_speed_corr": m["sync_speed_corr"],
            "move_fraction": m["move_fraction"],
            "turn_fraction": m["turn_fraction"],
            "usability": m["usability"],
            "s_sha256": _sha256(m["s_path"]),
            "v_sha256": _sha256(m["v_path"]),
            "s_path": str(Path(m["s_path"]).relative_to(Path(iovnbd_root))),
            "v_path": str(Path(m["v_path"]).relative_to(Path(iovnbd_root))),
            "load_warnings": [w for w in m["load_warnings"] if "using position" not in w],
        })

    by_driver: dict[str, float] = {}
    by_usability: dict[str, float] = {}
    for r in records:
        by_driver[r["driver"]] = by_driver.get(r["driver"], 0.0) + r["duration_s"]
        by_usability[r["usability"]] = by_usability.get(r["usability"], 0.0) + r["duration_s"]

    index = {
        "dataset": "IO-VNBD synchronised (Categorised)",
        "source": "github.com/onyekpeu/IO-VNBD @ master",
        "sample_rate_hz": 10,
        "n_sequences": len(records),
        "total_hours": round(sum(r["duration_s"] for r in records) / 3600, 2),
        "hours_by_driver": {k: round(v / 3600, 2) for k, v in sorted(by_driver.items())},
        "hours_by_usability": {k: round(v / 3600, 2) for k, v in sorted(by_usability.items())},
        "sequences": records,
    }
    out_path = Path(out_path)
    out_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iovnbd-root", default=_DEFAULT_ROOT)
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    args = ap.parse_args()
    idx = build(args.iovnbd_root, args.out)
    print(f"wrote {args.out}")
    print(f"  {idx['n_sequences']} sequences, {idx['total_hours']} h")
    print(f"  by driver:    {idx['hours_by_driver']}")
    print(f"  by usability: {idx['hours_by_usability']}")


if __name__ == "__main__":
    main()
