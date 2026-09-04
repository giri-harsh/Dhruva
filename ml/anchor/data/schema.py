"""Canonical column names for IO-VNBD's two CSV families, and the mapping from
the dataset's own (irregularly-spaced, mojibake, occasionally mislabelled)
headers to them.

Header facts (measured 2026-09-03 across all 144 synchronised CSVs — headers are
identical across every file in each family):

  * files are cp1252, not UTF-8 ("m/s²" -> "m/s\xb2", "µT" -> "\xb5T")
  * header has mixed ", " / "," separators and trailing spaces
    ("ORIENTATION (Roll ) (°)")
  * some unit labels are WRONG: vehicle "Height (km)" is metres; several
    "(0 or 1)" columns are actually 0-100 (%) — see notes below.

Canonical names are snake_case with an explicit unit suffix, in the ONE project
unit system (contracts/units.md). Raw -> canonical is by POSITION where headers
are unreliable, cross-checked against the normalised header text.
"""
from __future__ import annotations

# --- smartphone S-*.csv : 24 columns, fixed order ---
SMARTPHONE_COLUMNS: list[tuple[str, str]] = [
    # (canonical_name, raw_header_normalised_lower)
    ("gps_lat_deg",          "gps latitude (degrees)"),
    ("gps_lon_deg",          "gps longitude (degrees)"),
    ("gps_alt_m",            "gps altitude (m)"),
    ("gps_speed_kmh",        "gps speed (kmh)"),          # -> gps_speed_mps at load
    ("gps_accuracy_m",       "gps accuracy (m)"),
    ("gps_orientation_deg",  "gps orientation (°)"),
    ("gps_sats_raw",         "gps satellites in range"),  # "16 / 18" string
    ("time_since_start_ms",  "time since start (ms)"),
    ("date_local_raw",       "date (yyyy-mo-dd hh-mi-ss_sss)"),  # LOCAL time string
    ("accel_x_mps2",         "accelerometer x (m/s²)"),   # includes gravity
    ("accel_y_mps2",         "accelerometer y (m/s²)"),
    ("accel_z_mps2",         "accelerometer z (m/s²)"),
    ("gravity_x_mps2",       "gravity x (m/s²)"),
    ("gravity_y_mps2",       "gravity y (m/s²)"),
    ("gravity_z_mps2",       "gravity z (m/s²)"),
    ("gyro_yaw_radps",       "gyroscope yaw (rad/s)"),    # phone gyro IS rad/s
    ("gyro_pitch_radps",     "gyroscope pitch (rad/s)"),
    ("gyro_roll_radps",      "gyroscope roll (rad/s)"),
    ("mag_x_ut",             "magnetic field x (µt)"),
    ("mag_y_ut",             "magnetic field y (µt)"),
    ("mag_z_ut",             "magnetic field z (µt)"),
    ("orient_yaw_deg",       "orientation (yaw) (°)"),
    ("orient_pitch_deg",     "orientation (pitch) (°)"),
    ("orient_roll_deg",      "orientation (roll ) (°)"),
]

# --- vehicle V-*.csv : 29 columns, fixed order (CAN bus + Racelogic VBOX) ---
VEHICLE_COLUMNS: list[tuple[str, str]] = [
    ("gps_sats",              "no of gps satellites available"),
    ("time_of_day_s",         "time since start of day (seconds)"),  # UTC seconds
    ("gt_lat_deg",            "latitude (degrees)"),     # VBOX = ground truth
    ("gt_lon_deg",            "longitude (degrees)"),
    ("speed_kmh",             "velocity (km/hr)"),       # -> speed_mps (VBOX GT speed)
    ("heading_deg",           "heading (degrees)"),
    ("height_m",              "height (km)"),            # label lies: values are METRES
    ("vertical_velocity_kmh", "vertical velocity (km/hr)"),
    ("sample_period_s",       "sample period (seconds)"),
    ("steering_angle_deg",    "steering angle (degrees)"),
    ("wheel_fl_radps",        "wheel speed front left (rad/sec)"),
    ("wheel_fr_radps",        "wheel speed front right (rad/sec)"),
    ("wheel_rl_radps",        "wheel speed rear left (rad/sec)"),
    ("wheel_rr_radps",        "wheel speed rear right (rad/sec)"),
    ("yaw_rate_degps",        "yaw rate (deg/sec)"),     # DEG/s -> yaw_rate_radps at load
    ("indicated_speed_kmh",   "indicated vehicle speed (km/hr)"),
    ("long_accel_g",          "indicated longitudinal acceleration (g)"),  # g -> m/s²
    ("lat_accel_g",           "indicated lateral acceleration (g)"),
    ("handbrake",             "handbrake (0 or 1)"),
    ("gear_requested",        "gear requested (number fof gear employed 1-5)"),
    ("gear",                  "gear (number fof gear employed 1-5)"),
    ("engine_rpm",            "engine speed (rev/min)"),
    ("coolant_temp_c",        "coolant temperature (degrees)"),
    ("clutch",                "clutch position (0 or 1)"),
    ("brake_pressure_psi",    "brake pressure (psi)"),
    ("brake_position",        "brake position (0 or 1)"),
    ("battery_v",             "battery voltage (volts)"),
    ("air_temp_c",            "air temperature (degrees)"),
    ("accel_pedal_pct",       "accelerator pedal position (0 or 1)"),  # label lies: 0-100 %
]

# --- reduced 18-column smartphone variant (most unsynchronised France S-T*
#     files): GPS + time/date + accel + gravity + gyro, no magnetometer, no
#     device-orientation angles. All 6 model-input channels are present.
SMARTPHONE_COLUMNS_18: list[tuple[str, str]] = [
    ("gps_lat_deg",          "gps latitude (degrees)"),
    ("gps_lon_deg",          "gps longitude (degrees)"),
    ("gps_alt_m",            "gps altitude (m)"),
    ("gps_speed_kmh",        "gps speed (kmh)"),
    ("gps_accuracy_m",       "gps accuracy (m)"),
    ("gps_orientation_deg",  "gps orientation (°)"),
    ("gps_sats_raw",         "satellites in range"),
    ("time_since_start_ms",  "time since start (ms)"),
    ("date_local_raw",       "date (yyyy-mo-dd hh-mi-ss_sss)"),
    ("accel_x_mps2",         "accelerometer x (m/s²)"),
    ("accel_y_mps2",         "accelerometer y (m/s²)"),
    ("accel_z_mps2",         "accelerometer z (m/s²)"),
    ("gravity_x_mps2",       "gravity x (m/s²)"),
    ("gravity_y_mps2",       "gravity y (m/s²)"),
    ("gravity_z_mps2",       "gravity z (m/s²)"),
    ("gyro_yaw_radps",       "gyroscope yaw (rad/s)"),
    ("gyro_pitch_radps",     "gyroscope pitch (rad/s)"),
    ("gyro_roll_radps",      "gyroscope roll (rad/s)"),
]

SMARTPHONE_CANONICAL = [c for c, _ in SMARTPHONE_COLUMNS]
VEHICLE_CANONICAL = [c for c, _ in VEHICLE_COLUMNS]

# channels that must be interpolated (continuous) vs held (discrete/categorical)
# when a sequence is resampled or gap-filled.
VEHICLE_DISCRETE = {
    "gps_sats", "handbrake", "gear_requested", "gear", "clutch",
    "brake_position", "gear",
}


def normalise_header(raw: str) -> str:
    """Lower-case, drop non-ASCII (the unit symbols °, ², µ — which differ
    between cp1252 and our source and carry no matching value since units are
    in the canonical name), collapse whitespace. For tolerant positional
    cross-checking against the *_COLUMNS tables."""
    ascii_only = raw.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.strip().lower().split())


def parse_sats_in_range(value: str):
    """'16 / 18' -> 16 (satellites used). Returns float('nan') on anything odd."""
    try:
        return float(str(value).split("/")[0].strip())
    except (ValueError, AttributeError, IndexError):
        return float("nan")
