"""Unit conversion at the IO-VNBD ingestion boundary.

`contracts/units.md` fixes ONE internal unit system for the whole project:
m/s^2, rad/s, uT, m/s, metres, radians (in math). Any value entering from a
raw dataset column in a different unit is converted HERE, once, at load, and
never carried further.

IO-VNBD's own columns are a minefield (measured 2026-09-03, see
ml/docs/IO-VNBD-verification.md):

  Smartphone S-*.csv
    ACCELEROMETER X/Y/Z   m/s^2      -> no conversion
    GRAVITY X/Y/Z         m/s^2      -> no conversion
    GYROSCOPE Yaw/Pitch/Roll  rad/s  -> NO conversion (phone gyro really is rad/s)
    MAGNETIC FIELD X/Y/Z  uT         -> no conversion
    GPS SPEED             km/h       -> / 3.6

  Vehicle V-*.csv  (CAN + VBOX)
    Wheel Speed FL/FR/RL/RR   rad/s  -> no conversion
    Yaw Rate                  DEG/s  -> * pi/180     <-- the units.md ~57x trap
    Indicated Long/Lat Accel  g      -> * 9.80665    <-- silent 10x error if missed
    Velocity / Indicated Vehicle Speed / Vertical velocity   km/h -> / 3.6
    Height              header says km, VALUES ARE METRES (71-148 in UK, GPS
                        altitude on the phone reads 126-196 m for the same
                        drives). The "(km)" label is wrong. -> NO conversion.

Angular-rate sanity (contracts/units.md): a deg/s->rad/s mistake is ~57x and
moves the WHOLE distribution. A phone gyro loose in a cup-holder can legitimately
spike past 10 rad/s on a pothole (measured: pitch to 12.7 rad/s) without any unit
error. So `assert_gyro_sane` checks a high PERCENTILE, not every sample:
p99.5(|omega|) < 10 rad/s catches a real unit error; rare legit spikes pass and
are clipped by the caller if needed. The CAN vehicle-frame yaw rate, after the
deg->rad conversion, is checked hard (it never legitimately exceeds ~1.5 rad/s).
"""
from __future__ import annotations

import numpy as np

# --- exact constants ---
KMH_TO_MPS = 1.0 / 3.6
G_TO_MPS2 = 9.80665          # standard gravity, exact (CGPM 1901)
DEG_TO_RAD = np.pi / 180.0
KM_TO_M = 1000.0

# --- sanity bounds (from contracts/units.md + physical plausibility) ---
GYRO_MAX_RADPS = 10.0        # ~573 deg/s; no road vehicle yaws this fast
ACCEL_MAX_MPS2 = 100.0       # ~10 g; a phone in a car never sees this sustained
WHEEL_OMEGA_MAX_RADPS = 400.0  # ~0.28 m radius * 400 = 112 m/s = 400 km/h; generous


class UnitSanityError(ValueError):
    """Raised when a converted channel is outside a physically possible range,
    which almost always means the source column was in a different unit than
    assumed. See contracts/units.md 'the one bug that will happen'."""


def kmh_to_mps(x):
    return np.asarray(x, dtype=np.float64) * KMH_TO_MPS


def g_to_mps2(x):
    return np.asarray(x, dtype=np.float64) * G_TO_MPS2


def deg_to_rad(x):
    return np.asarray(x, dtype=np.float64) * DEG_TO_RAD


def km_to_m(x):
    return np.asarray(x, dtype=np.float64) * KM_TO_M


def assert_gyro_sane(values, name: str, *, hard: bool = False, percentile: float = 99.5) -> int:
    """contracts/units.md's defensive rule for angular-rate channels, applied
    AFTER converting to rad/s. Returns the count of samples that individually
    exceed GYRO_MAX_RADPS (so the caller can clip + log them).

    hard=False (phone 3-axis gyro, device frame): a real deg/s->rad/s error is
      ~57x and shifts the whole distribution, so we test the `percentile` value,
      not the max. Rare pothole spikes above 10 rad/s do not trip it.
    hard=True (CAN vehicle-frame yaw rate): test the max — this channel never
      legitimately exceeds ~1.5 rad/s, so any sample past 10 is a unit error.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return 0
    absv = np.abs(finite)
    peak = float(absv.max())
    gate = peak if hard else float(np.percentile(absv, percentile))
    if gate > GYRO_MAX_RADPS:
        raise UnitSanityError(
            f"angular-rate channel '{name}': "
            f"{'max' if hard else f'p{percentile}'}(|value|) = {gate:.2f} rad/s "
            f"(> {GYRO_MAX_RADPS} ~ 573 deg/s). This almost certainly means the "
            f"source column was deg/s and the * pi/180 conversion was missed "
            f"(see contracts/units.md — the ~57x bug)."
        )
    return int((absv > GYRO_MAX_RADPS).sum())


def assert_accel_sane(values, name: str) -> None:
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return
    peak = float(np.abs(finite).max())
    if peak > ACCEL_MAX_MPS2:
        raise UnitSanityError(
            f"acceleration channel '{name}': |value| max = {peak:.1f} m/s^2 "
            f"(> {ACCEL_MAX_MPS2} ~ 10 g). Likely a unit error — CAN accel "
            f"columns are in g and need * 9.80665 (see contracts/units.md)."
        )


def assert_wheel_omega_sane(values, name: str) -> None:
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return
    peak = float(np.abs(finite).max())
    if peak > WHEEL_OMEGA_MAX_RADPS:
        raise UnitSanityError(
            f"wheel-speed channel '{name}': |value| max = {peak:.1f} rad/s "
            f"(> {WHEEL_OMEGA_MAX_RADPS}). Expected rad/s (~40 at 40 km/h)."
        )
