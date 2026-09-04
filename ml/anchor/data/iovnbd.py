"""Load a single IO-VNBD CSV (smartphone or vehicle) into a canonical DataFrame.

Everything unit-related happens HERE, at the boundary (contracts/units.md):
raw column -> canonical name -> project unit -> sanity assertion. Nothing
downstream ever sees km/h, deg/s, or g.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import units
from .schema import (
    SMARTPHONE_COLUMNS,
    SMARTPHONE_COLUMNS_18,
    VEHICLE_COLUMNS,
    normalise_header,
    parse_sats_in_range,
)

CP1252 = "cp1252"


class HeaderMismatchError(ValueError):
    """Raised when a CSV's header does not match the frozen column list —
    a new dataset revision or a wrong file, either way stop, don't guess."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    path: Path
    kind: str  # "smartphone" | "vehicle"
    n_rows: int
    warnings: list[str] = field(default_factory=list)


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=CP1252, skipinitialspace=True, dtype=str,
                     keep_default_na=True, na_values=["", "NaN", "nan"])
    df.columns = [normalise_header(c) for c in df.columns]
    # some unsynchronised exports have a trailing comma on every row -> a phantom
    # empty last column ("unnamed: N"); drop it if it's entirely empty.
    while df.columns[-1].startswith("unnamed") and df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]
    return df


def _check_header(df: pd.DataFrame, spec: list[tuple[str, str]], path: Path) -> list[str]:
    warnings: list[str] = []
    expected = [normalise_header(raw) for _, raw in spec]
    got = list(df.columns)
    if len(got) != len(expected):
        raise HeaderMismatchError(
            f"{path.name}: expected {len(expected)} columns, got {len(got)}.\n"
            f"  expected: {expected}\n  got:      {got}"
        )
    for i, (exp, g) in enumerate(zip(expected, got)):
        if exp != g:
            # positional trust wins, but record every text disagreement
            warnings.append(f"col {i}: header text '{g}' != expected '{exp}' "
                            f"(using position)")
    return warnings


def _to_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def load_smartphone_csv(path: str | Path) -> LoadResult:
    path = Path(path)
    raw = _read_raw(path)
    if len(raw.columns) == len(SMARTPHONE_COLUMNS_18):
        spec = SMARTPHONE_COLUMNS_18
    else:
        spec = SMARTPHONE_COLUMNS
    warnings = _check_header(raw, spec, path)
    reduced = spec is SMARTPHONE_COLUMNS_18

    canon = [c for c, _ in spec]
    src = {canon[i]: raw.iloc[:, i] for i in range(len(canon))}
    out = pd.DataFrame(index=raw.index)

    passthrough = ["gps_lat_deg", "gps_lon_deg", "gps_alt_m", "gps_accuracy_m",
                   "gps_orientation_deg",
                   "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
                   "gravity_x_mps2", "gravity_y_mps2", "gravity_z_mps2",
                   "gyro_yaw_radps", "gyro_pitch_radps", "gyro_roll_radps"]
    if not reduced:
        passthrough += ["mag_x_ut", "mag_y_ut", "mag_z_ut",
                        "orient_yaw_deg", "orient_pitch_deg", "orient_roll_deg"]
    for name in passthrough:
        out[name] = _to_float(src[name])
    if reduced:  # keep the schema uniform for downstream — fill absent channels
        for name in ["mag_x_ut", "mag_y_ut", "mag_z_ut",
                     "orient_yaw_deg", "orient_pitch_deg", "orient_roll_deg"]:
            out[name] = np.nan
        warnings.append("reduced 18-col schema: magnetometer + orientation-angle "
                        "channels absent (filled NaN)")

    # conversions at the boundary
    out["gps_speed_mps"] = units.kmh_to_mps(_to_float(src["gps_speed_kmh"]))

    # bespoke parses
    out["time_since_start_ms"] = _to_float(src["time_since_start_ms"])
    out["gps_sats"] = src["gps_sats_raw"].map(parse_sats_in_range).astype(float)
    out["date_local_raw"] = src["date_local_raw"].astype("string")

    # --- sanity (contracts/units.md), AFTER conversion ---
    for ch in ["gyro_yaw_radps", "gyro_pitch_radps", "gyro_roll_radps"]:
        n_spike = units.assert_gyro_sane(out[ch].to_numpy(), f"{path.name}:{ch}", hard=False)
        if n_spike:
            warnings.append(f"{ch}: {n_spike} samples > {units.GYRO_MAX_RADPS} rad/s "
                            f"(legit spikes, left as-is)")
    for ch in ["accel_x_mps2", "accel_y_mps2", "accel_z_mps2"]:
        units.assert_accel_sane(out[ch].to_numpy(), f"{path.name}:{ch}")

    return LoadResult(df=out, path=path, kind="smartphone", n_rows=len(out), warnings=warnings)


def load_vehicle_csv(path: str | Path) -> LoadResult:
    path = Path(path)
    raw = _read_raw(path)
    warnings = _check_header(raw, VEHICLE_COLUMNS, path)

    canon = [c for c, _ in VEHICLE_COLUMNS]
    src = {canon[i]: raw.iloc[:, i] for i in range(len(canon))}
    out = pd.DataFrame(index=raw.index)

    for name in ["gps_sats", "time_of_day_s", "gt_lat_deg", "gt_lon_deg",
                 "heading_deg", "sample_period_s", "steering_angle_deg",
                 "wheel_fl_radps", "wheel_fr_radps", "wheel_rl_radps", "wheel_rr_radps",
                 "handbrake", "gear_requested", "gear", "engine_rpm",
                 "coolant_temp_c", "clutch", "brake_pressure_psi", "brake_position",
                 "battery_v", "air_temp_c", "accel_pedal_pct"]:
        out[name] = _to_float(src[name])

    # "Height (km)" — the label is wrong, values are metres (see schema.py). No convert.
    out["height_m"] = _to_float(src["height_m"])

    # conversions at the boundary
    out["speed_mps"] = units.kmh_to_mps(_to_float(src["speed_kmh"]))
    out["indicated_speed_mps"] = units.kmh_to_mps(_to_float(src["indicated_speed_kmh"]))
    out["vertical_velocity_mps"] = units.kmh_to_mps(_to_float(src["vertical_velocity_kmh"]))
    out["yaw_rate_radps"] = units.deg_to_rad(_to_float(src["yaw_rate_degps"]))
    out["long_accel_mps2"] = units.g_to_mps2(_to_float(src["long_accel_g"]))
    out["lat_accel_mps2"] = units.g_to_mps2(_to_float(src["lat_accel_g"]))

    # --- sanity ---
    # CAN vehicle-frame yaw rate: hard check (never legitimately > ~1.5 rad/s)
    units.assert_gyro_sane(out["yaw_rate_radps"].to_numpy(), f"{path.name}:yaw_rate_radps", hard=True)
    for ch in ["wheel_fl_radps", "wheel_fr_radps", "wheel_rl_radps", "wheel_rr_radps"]:
        units.assert_wheel_omega_sane(out[ch].to_numpy(), f"{path.name}:{ch}")
    for ch in ["long_accel_mps2", "lat_accel_mps2"]:
        units.assert_accel_sane(out[ch].to_numpy(), f"{path.name}:{ch}")

    return LoadResult(df=out, path=path, kind="vehicle", n_rows=len(out), warnings=warnings)
