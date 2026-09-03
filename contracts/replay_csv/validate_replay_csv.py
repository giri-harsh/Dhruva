"""
Validates a replay CSV against contracts/replay_csv/schema.json.

Run this on ANY csv before handing it to the Android side, and as a CI
check. Exits non-zero with a specific, actionable message on failure —
never a silent pass on a subtly wrong file.

Usage:
    python validate_replay_csv.py path/to/file.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SCHEMA = json.loads((HERE / "schema.json").read_text())
EXPECTED_COLUMNS = [c["name"] for c in SCHEMA["columns"]]
COLUMN_INFO = {c["name"]: c for c in SCHEMA["columns"]}


def fail(msg: str) -> "NoReturn":
    print(f"FAIL: {msg}")
    sys.exit(1)


def validate(path: Path) -> None:
    raw = path.read_bytes()

    if raw.startswith(b"\xef\xbb\xbf"):
        fail(f"{path}: has a UTF-8 BOM. Re-save without BOM (encoding must be plain UTF-8).")

    if b"\r\n" in raw or b"\r" in raw:
        fail(f"{path}: contains CR bytes (CRLF line endings). Must be LF only — "
             f"check .gitattributes is applied, or the file was hand-edited on Windows "
             f"without LF enforcement.")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    if list(df.columns) != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        order_msg = ""
        if not missing and not extra and list(df.columns) != EXPECTED_COLUMNS:
            order_msg = " Columns present but in the WRONG ORDER."
        fail(
            f"{path}: column mismatch.{order_msg}\n"
            f"  expected: {EXPECTED_COLUMNS}\n"
            f"  got:      {list(df.columns)}\n"
            f"  missing:  {sorted(missing)}\n"
            f"  extra:    {sorted(extra)}"
        )

    for col in EXPECTED_COLUMNS:
        info = COLUMN_INFO[col]
        series = df[col]

        for i, raw_val in series.items():
            if raw_val == "":
                if "or empty" not in info["dtype"] and col != "gnss_valid":
                    fail(f"{path}: row {i}, column '{col}' is empty but this column is not "
                         f"nullable per schema (dtype={info['dtype']}).")
                continue
            if "," in raw_val:
                fail(f"{path}: row {i}, column '{col}' value '{raw_val}' contains a comma — "
                     f"likely a locale decimal-separator bug (comma instead of '.'). "
                     f"See known_failure_modes in schema.json.")
            try:
                float(raw_val)
            except ValueError:
                fail(f"{path}: row {i}, column '{col}' value '{raw_val}' is not numeric.")

        # gnss_valid must be exactly 0 or 1
        if col == "gnss_valid":
            bad = series[~series.isin(["0", "1"])]
            if len(bad):
                fail(f"{path}: gnss_valid must be 0 or 1, found other values at rows {list(bad.index)}")

    # Cross-field rule: when gnss_valid == 0, the four gnss_* fields must be empty
    gnss_fields = ["gnss_lat", "gnss_lon", "gnss_speed_mps", "gnss_course_deg"]
    invalid_rows = df[df["gnss_valid"] == "0"]
    for col in gnss_fields:
        nonempty = invalid_rows[invalid_rows[col] != ""]
        if len(nonempty):
            fail(f"{path}: rows {list(nonempty.index)} have gnss_valid=0 but a non-empty "
                 f"'{col}' — GNSS-invalid rows must leave all gnss_* fields empty, not 0.")

    # Sanity check: gyro magnitude. If someone accidentally logged deg/s instead
    # of rad/s, values will be ~57x too large and this catches it immediately
    # instead of letting it silently diverge the filter later.
    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) and vals.abs().max() > 10.0:
            fail(f"{path}: column '{col}' has |value| > 10 rad/s (max={vals.abs().max():.2f}). "
                 f"A car does not yaw at >10 rad/s (~573 deg/s). This almost certainly means "
                 f"gyro was logged in deg/s, not rad/s. Convert with value * pi / 180.")

    # Sanity check: timestamps strictly increasing
    ts = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    if not ts.is_monotonic_increasing:
        fail(f"{path}: timestamp_ms is not strictly increasing throughout the file.")

    print(f"OK: {path} — {len(df)} rows, schema v{SCHEMA['contract_version']} satisfied.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python validate_replay_csv.py path/to/file.csv")
        sys.exit(2)
    validate(Path(sys.argv[1]))
