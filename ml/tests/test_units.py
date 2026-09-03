import numpy as np
import pytest

from anchor.data import units


def test_kmh_to_mps():
    assert units.kmh_to_mps(36.0) == pytest.approx(10.0)
    assert units.kmh_to_mps(np.array([0.0, 3.6, 36.0])).tolist() == pytest.approx([0.0, 1.0, 10.0])


def test_g_to_mps2():
    assert units.g_to_mps2(1.0) == pytest.approx(9.80665)


def test_deg_to_rad():
    assert units.deg_to_rad(180.0) == pytest.approx(np.pi)


def test_gyro_sane_passes_real_data():
    # a normal drive: mostly < 1 rad/s, a couple of pothole spikes to ~12
    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.3, 10_000)
    v[500] = 12.6
    v[7001] = -11.1
    n_spike = units.assert_gyro_sane(v, "phone_gyro", hard=False)
    assert n_spike == 2  # reported for clipping, not raised


def test_gyro_sane_catches_degps_error_soft():
    # deg/s mistaken for rad/s: whole distribution ~57x too big
    rng = np.random.default_rng(1)
    v_radps = rng.normal(0, 0.3, 10_000)
    v_as_degps = v_radps * 180.0 / np.pi
    with pytest.raises(units.UnitSanityError):
        units.assert_gyro_sane(v_as_degps, "bad", hard=False)


def test_gyro_sane_hard_catches_single_bad_sample():
    v = np.zeros(1000)
    v[3] = 30.0  # one impossible sample on a vehicle-frame yaw-rate channel
    with pytest.raises(units.UnitSanityError):
        units.assert_gyro_sane(v, "can_yaw", hard=True)


def test_accel_sane_catches_impossible_magnitude():
    # a channel peaking at ~15 g sustained is not a phone in a car — likely a
    # double conversion or a wrong column entirely.
    v = np.full(1000, 12.0) * 9.80665  # ~118 m/s^2
    with pytest.raises(units.UnitSanityError):
        units.assert_accel_sane(v, "impossible")


def test_accel_sane_passes_normal_phone_data():
    rng = np.random.default_rng(2)
    v = 9.81 + rng.normal(0, 2.0, 10_000)  # gravity + road, occasional bump
    v[10] = 55.0  # a hard bump, ~5.6 g — still physical
    units.assert_accel_sane(v, "ok")  # must not raise
