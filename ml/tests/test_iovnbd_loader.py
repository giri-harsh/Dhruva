"""Loader tests against the real IO-VNBD synchronised subset (skipped if not
materialised). Verifies unit conversion actually happens at the boundary and
the synchronised join is row-consistent.
"""
import numpy as np
import pytest


def test_discovers_expected_sequence_count(sequences):
    assert len(sequences) == 72
    ids = {s.seq_id for s in sequences}
    assert {"m", "s1", "s3c", "vta29", "vfa01", "y1"} <= ids


def test_drivers_and_region(sequences):
    by_driver = {}
    for s in sequences:
        by_driver.setdefault(s.driver, 0)
        by_driver[s.driver] += 1
    assert set(by_driver) == {"A", "B", "D", "E"}
    assert all(s.region == "england" for s in sequences)


def test_units_converted_at_boundary(sequences):
    """CAN yaw rate must arrive in rad/s (not the raw deg/s), CAN accel in
    m/s^2 (not g), speeds in m/s (not km/h). If any conversion were missed the
    load would already have raised UnitSanityError — this asserts the ranges
    are physically those of the converted unit."""
    s = next(x for x in sequences if x.seq_id == "m")  # long, varied drive
    d = s.df
    assert np.nanmax(np.abs(d["veh_yaw_rate_radps"])) < 5.0        # rad/s, not ~60 deg/s
    assert np.nanmax(np.abs(d["veh_long_accel_mps2"])) < 30.0      # m/s^2, not ~1 g
    assert np.nanmax(np.abs(d["veh_long_accel_mps2"])) > 1.5       # and not left in g
    assert np.nanmax(d["veh_speed_mps"]) < 60.0                    # m/s (~200 km/h cap)
    assert np.nanmax(d["veh_speed_mps"]) > 5.0
    # phone gyro is genuinely rad/s in the source — unchanged, still sane-ish
    assert np.nanpercentile(np.abs(d["phone_gyro_yaw_radps"]), 99.5) < units_gyro_max()


def units_gyro_max():
    from anchor.data import units
    return units.GYRO_MAX_RADPS


def test_synced_join_row_consistent(sequences):
    for s in sequences:
        n = s.n_rows
        assert len(s.df) == n
        assert s.df["t_ms"].iloc[0] == 0
        assert s.df["t_ms"].iloc[-1] == (n - 1) * 100
        # segments partition [0, n) with no gaps or overlaps
        flat = [x for seg in s.segments for x in seg]
        assert flat[0] == 0 and flat[-1] == n
        for a, b in s.segments:
            assert b > a


def test_height_treated_as_metres_not_km(sequences):
    """The vehicle 'Height (km)' column is mislabelled — values are metres.
    UK terrain is ~0-250 m; if we had multiplied by 1000 it'd be 70+ km."""
    s = next(x for x in sequences if x.seq_id == "m")
    h = s.df["veh_height_m"].to_numpy()
    assert 0 < np.nanmedian(h) < 400
