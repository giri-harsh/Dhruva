"""Generates the literal sample_replay.csv fixture from schema.json's column
list, so the fixture can never drift from the schema by hand-editing either
one separately. Also produces a few rows with GNSS present and a few with
GNSS absent (outage), matching how real replay files will look.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
schema = json.loads((HERE / "schema.json").read_text())
columns = [c["name"] for c in schema["columns"]]

rows = []
t0 = 1_735_000_000_000  # arbitrary fixed epoch ms, deterministic fixture
for i in range(30):
    t = t0 + i * 100  # 10 Hz
    gnss_present = i < 10 or i >= 20  # outage between sample 10 and 20
    row = {
        "timestamp_ms": t,
        "accel_x": round(0.05 * (i % 5), 4),
        "accel_y": round(-0.02 * (i % 3), 4),
        "accel_z": round(9.81 + 0.01 * (i % 2), 4),
        "gyro_x": round(0.001 * (i % 4), 5),
        "gyro_y": round(-0.0005 * (i % 3), 5),
        "gyro_z": round(0.002 * (i % 6), 5),
        "mag_x": round(20.0 + 0.1 * i, 3),
        "mag_y": round(-5.0 + 0.05 * i, 3),
        "mag_z": round(40.0 - 0.02 * i, 3),
        "gnss_valid": 1 if gnss_present else 0,
        "gnss_lat": round(28.6139 + 0.0001 * i, 7) if gnss_present else "",
        "gnss_lon": round(77.2090 + 0.0001 * i, 7) if gnss_present else "",
        "gnss_speed_mps": round(12.0 + 0.1 * i, 3) if gnss_present else "",
        "gnss_course_deg": round(45.0 + 0.5 * i, 2) if gnss_present else "",
    }
    rows.append(row)

out_path = HERE / "sample_replay.csv"
with out_path.open("w", newline="\n", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {out_path} ({len(rows)} rows, GNSS outage rows 10-19 inclusive)")
