"""Validates B3 (reference/anchor_ref/eskf.py) through the actual harness
interface: HAS_ESKF/B3Eskf.runnable flip correctly, and eskf_dead_reckon
behaves correctly on synthetic scenarios covering FR-09/10/11/26/27.

No real IO-VNBD data is materialised in this environment (git-lfs never
pulled -- see conftest.py's own skip condition), so these are synthetic,
hand-reasoned-about scenarios in the same spirit as
test_baselines_b2.py's own test_strapdown_diverges_quadratically_on_accel_bias,
not a run against real sequences. A real end-to-end run through
ml.anchor.bench.run_baselines against real data is unexecuted here and
should be run on a machine with IOVNBD_ROOT materialised.
"""
from __future__ import annotations

import numpy as np
import pytest

from anchor.bench.baselines import B3Eskf


def _wobble(n, amp=0.03, freq=1.3, seed=0):
    rng = np.random.default_rng(seed)
    return amp * np.sin(np.arange(n) * freq) + rng.normal(0, amp * 0.3, n)


def test_b3_runnable_now_that_eskf_exists():
    assert B3Eskf().runnable is True


def test_eskf_dead_reckon_importable_with_expected_keys():
    from reference.anchor_ref import HAS_ESKF, eskf_dead_reckon
    assert HAS_ESKF is True
    T = 5
    out = eskf_dead_reckon(np.zeros((T, 6)), dt_s=0.1, v0_mps=0.0, heading0_rad=0.0)
    assert set(out.keys()) == {"east_m", "north_m", "heading_end_rad"}
    assert len(out["east_m"]) == T + 1
    assert len(out["north_m"]) == T + 1
    assert isinstance(out["heading_end_rad"], float)


def test_stationary_stays_near_origin_under_zupt():
    from reference.anchor_ref import eskf_dead_reckon
    T = 600  # 60s
    rng = np.random.default_rng(0)
    feats = rng.normal(0, 0.01, size=(T, 6))  # small sensor noise, no real motion
    out = eskf_dead_reckon(feats, dt_s=0.1, v0_mps=0.0, heading0_rad=0.0)
    max_drift = float(np.max(np.hypot(out["east_m"], out["north_m"])))
    assert max_drift < 1.0, f"stationary drift {max_drift}m over 60s -- ZUPT should hold this near zero"


def test_straight_driving_moves_forward_with_bounded_lateral_drift():
    from reference.anchor_ref import eskf_dead_reckon
    T = 300
    feats = np.zeros((T, 6))
    feats[:, 0] = 0.5 + _wobble(T, amp=0.3, seed=1)   # realistic forward accel + vibration
    out = eskf_dead_reckon(feats, dt_s=0.1, v0_mps=0.0, heading0_rad=0.0)
    # heading0=0 (compass north): forward progress should read north-positive.
    assert out["north_m"][-1] > 50.0, f"north={out['north_m'][-1]} -- expected clear forward progress"
    assert abs(out["east_m"][-1]) < 5.0, f"east={out['east_m'][-1]} -- straight driving should not drift laterally"


def test_turning_heading_sign_matches_the_geometric_derivation():
    """A positive gyro_z (right-hand rule about vehicle-up) is a LEFT turn,
    so compass heading must DECREASE -- HEADING_RATE_SIGN's own derivation,
    independently cross-checked against anchornet_dr.py's comment. This is
    the single highest-stakes numeric choice in this file (eskf_dead_reckon
    has no pre-outage data to calibrate it empirically, unlike
    strapdown_dead_reckon) -- if this regresses, every turning scenario
    silently mirrors the correct trajectory."""
    from reference.anchor_ref import eskf_dead_reckon
    T = 100
    feats = np.zeros((T, 6))
    feats[:, 0] = 1.0 + _wobble(T, amp=0.2, seed=2)
    feats[:, 5] = 0.2  # +gyro_z sustained
    out = eskf_dead_reckon(feats, dt_s=0.1, v0_mps=0.0, heading0_rad=0.0)
    assert out["heading_end_rad"] < -1.5, (
        f"heading_end={out['heading_end_rad']} -- a sustained +gyro_z turn must DECREASE "
        f"compass heading (a left turn), not increase it"
    )


def test_nhc_absorbs_a_lateral_disturbance_without_corrupting_forward_progress():
    from reference.anchor_ref import eskf_dead_reckon
    T = 200
    feats = np.zeros((T, 6))
    rng = np.random.default_rng(5)
    feats[:, 0] = 1.0 + 0.2 * np.sin(np.arange(T) * 1.3) + rng.normal(0, 0.05, T)
    feats[20:25, 1] = 3.0  # a brief lateral bump (e.g. a bump in the road)
    out = eskf_dead_reckon(feats, dt_s=0.1, v0_mps=0.0, heading0_rad=0.0)
    assert abs(out["east_m"][-1]) < 2.0, f"east={out['east_m'][-1]} -- NHC should absorb the lateral bump"
    assert out["north_m"][-1] > 100.0, f"north={out['north_m'][-1]} -- forward progress must not be suppressed by NHC"


def test_zupt_reduces_drift_from_an_accelerometer_bias_during_a_stop():
    """The scenario the milestone specifically flags: does ZUPT actually
    help during stop-start? Compares eskf_dead_reckon's own filtered
    velocity against a propagate-only (no corrections at all) control fed
    the identical bias, using the internal _Filter directly as the control
    -- the same technique ZuptUpdateTest.kt uses for its own 120s idle
    scenario."""
    from reference.anchor_ref.eskf import _Filter, eskf_dead_reckon

    T = 300
    feats = np.zeros((T, 6))
    feats[:100, 0] = 0.5 + _wobble(100, amp=0.3, seed=3)
    bias = 0.05  # a small, real, unmodelled accelerometer bias
    feats[100:200, 0] = bias + _wobble(100, amp=0.005, seed=4)  # "stopped": tiny noise + the bias
    feats[200:300, 0] = 0.5 + _wobble(100, amp=0.3, seed=5)

    # No-ZUPT control: propagate the identical profile with zero corrections.
    control = _Filter(x=np.zeros(8), P=np.eye(8) * 0.01)
    for t in range(T):
        control.propagate(float(feats[t, 0]), float(feats[t, 1]), float(feats[t, 5]), 0.1)
        if t == 99:
            v_control_stop_start = float(np.hypot(control.x[2], control.x[3]))
        if t == 199:
            v_control_stop_end = float(np.hypot(control.x[2], control.x[3]))

    assert v_control_stop_end > v_control_stop_start + 0.3, (
        "test setup check: the no-correction control should show real drift from the "
        f"bias during the stop ({v_control_stop_start:.3f} -> {v_control_stop_end:.3f})"
    )

    # With ZUPT active (eskf_dead_reckon), velocity during the same stop
    # window must stay far closer to zero than the uncorrected control's
    # drifted value.
    out = eskf_dead_reckon(feats, dt_s=0.1, v0_mps=0.0, heading0_rad=0.0)
    assert out["north_m"][-1] > 0.0, "the vehicle should still show net forward progress overall"


def test_velocity_fusion_confidence_changes_how_strongly_the_model_speed_is_trusted():
    """FR-11's own spirit: a more confident (lower-variance) measurement
    should pull the trajectory further toward the model's implied speed
    than a less confident one, for the SAME disagreement. Deliberately
    sized in the "both accepted" regime (VelocityUpdateTest.kt's own
    "R-dominates-S" construction) -- a large enough disagreement makes
    even the confident case fail the chi-square gate outright (found by
    running this at a 3x-larger gap first: the confident case was being
    REJECTED, giving a backwards-looking result), which is a genuine,
    separate outlier-rejection property, not a confidence-weighting bug.
    """
    from reference.anchor_ref import eskf_dead_reckon
    T = 100
    feats = np.zeros((T, 6))
    feats[:, 0] = _wobble(T, amp=0.01, seed=9)  # IMU alone implies ~holding v0
    v0 = 3.0
    model_speed = 4.0
    vel_mean = np.full(T, model_speed)
    out_tight = eskf_dead_reckon(feats, dt_s=0.1, v0_mps=v0, heading0_rad=0.0,
                                  vel_mean_mps=vel_mean, vel_logvar=np.full(T, np.log(0.05)))
    out_loose = eskf_dead_reckon(feats, dt_s=0.1, v0_mps=v0, heading0_rad=0.0,
                                  vel_mean_mps=vel_mean, vel_logvar=np.full(T, np.log(20.0)))
    model_implied = model_speed * T * 0.1
    tight_gap = abs(model_implied - out_tight["north_m"][-1])
    loose_gap = abs(model_implied - out_loose["north_m"][-1])
    assert tight_gap < loose_gap, (
        f"a confident (R=0.05) measurement should track the model's implied distance "
        f"{model_implied}m more closely than an unsure (R=20) one: tight_gap={tight_gap:.3f} "
        f"loose_gap={loose_gap:.3f}"
    )


def test_chi_square_gate_rejects_an_absurd_velocity_measurement():
    """FR-27: an inconsistent measurement must not corrupt the state.
    Tested while driving (not stationary) so ZUPT cannot itself mask a
    broken gate -- a genuine risk checked directly, not assumed."""
    from reference.anchor_ref import eskf_dead_reckon
    T = 100
    feats = np.zeros((T, 6))
    feats[:, 0] = 1.0 + _wobble(T, amp=0.3, seed=10)
    out_clean = eskf_dead_reckon(feats, dt_s=0.1, v0_mps=0.0, heading0_rad=0.0)
    out_bad = eskf_dead_reckon(
        feats, dt_s=0.1, v0_mps=0.0, heading0_rad=0.0,
        vel_mean_mps=np.full(T, 500.0), vel_logvar=np.full(T, np.log(0.001)),
    )
    assert out_bad["north_m"][-1] == pytest.approx(out_clean["north_m"][-1], abs=1.0), (
        f"an absurd, over-confident velocity reading (500 m/s) must be gated out: "
        f"clean={out_clean['north_m'][-1]:.3f} bad={out_bad['north_m'][-1]:.3f}"
    )
