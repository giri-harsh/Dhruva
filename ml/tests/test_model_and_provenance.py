import numpy as np
import pytest
import torch

from anchor.contract import NUM_FEATURES, WINDOW_SIZE_SAMPLES
from anchor.models.anchornet import AnchorNet, AnchorNetConfig


def test_model_matches_contract_io():
    m = AnchorNet()
    x = torch.randn(8, WINDOW_SIZE_SAMPLES, NUM_FEATURES)
    out = m(x)
    assert set(out) == {"velocity_mean_mps", "velocity_log_variance"}
    assert out["velocity_mean_mps"].shape == (8, 1)
    assert out["velocity_log_variance"].shape == (8, 1)
    # Head A is softplus -> mean speed strictly >= 0
    assert torch.all(out["velocity_mean_mps"] >= 0)


def test_model_rejects_wrong_input_shape():
    m = AnchorNet()
    with pytest.raises(ValueError):
        m(torch.randn(4, WINDOW_SIZE_SAMPLES, NUM_FEATURES + 1))
    with pytest.raises(ValueError):
        m(torch.randn(4, WINDOW_SIZE_SAMPLES + 5, NUM_FEATURES))


def test_param_budget_under_50k():
    assert AnchorNet(AnchorNetConfig(trunk="tcn")).num_parameters() < 50_000
    assert AnchorNet(AnchorNetConfig(enable_context_head=True, enable_yaw_head=True)
                     ).num_parameters() < 50_000


def test_tcn_receptive_field_is_15():
    """Perturbing an input sample should not change an output more than
    ceil(RF/2) = 8 steps away (a cheap RF check)."""
    m = AnchorNet(AnchorNetConfig(dropout=0.0)).eval()
    # feed a per-timestep impulse and see how far the pre-pool activation spreads
    trunk = m.trunk
    x0 = torch.zeros(1, WINDOW_SIZE_SAMPLES, NUM_FEATURES)
    h0 = _trunk_time_activation(trunk, x0)
    x1 = x0.clone(); x1[0, WINDOW_SIZE_SAMPLES // 2, 0] = 1.0
    h1 = _trunk_time_activation(trunk, x1)
    influenced = (h1 - h0).abs().sum(0) > 1e-6
    idx = torch.nonzero(influenced).flatten()
    spread = int(idx.max() - idx.min()) + 1 if len(idx) else 0
    assert spread <= 15, f"receptive field {spread} > 15"


def _trunk_time_activation(trunk, x):
    h = trunk.inproj(x.transpose(1, 2))
    for blk in trunk.blocks:
        h = blk(h)
    return h[0]  # [H, T]


# ----------------------------- FR-07 provenance -----------------------------

@pytest.mark.usefixtures("sequences")
def test_fr07_no_gnss_or_can_column_reaches_the_model_tensor(sequences):
    """FR-07 acceptance: the inference feature tensor is derived from phone IMU
    ONLY. Zeroing every vehicle (CAN/VBOX) and phone-GNSS column must not change
    the model input features, given a fixed alignment. (The per-sequence yaw
    calibration may consult CAN longitudinal accel per frame_convention.md, but
    that produces a fixed rotation, not a per-window data path — so we hold the
    alignment fixed and prove the window features carry no vehicle data.)"""
    from anchor.data.features import align_sequence_to_vehicle_frame, sequence_model_features

    seq = next(s for s in sequences if s.meta["usability"] == "use" and s.n_rows > 4000)
    align = align_sequence_to_vehicle_frame(seq)          # fixed R
    base = sequence_model_features(seq, align)

    poisoned = seq
    df = seq.df.copy()
    for col in df.columns:
        if col.startswith("veh_") or col.startswith("phone_gps_"):
            df[col] = np.random.default_rng(0).normal(size=len(df))
    poisoned.df = df
    after = sequence_model_features(poisoned, align)

    assert np.allclose(base, after, atol=1e-6), (
        "model input features changed when CAN/GNSS columns were randomised — "
        "a vehicle/GNSS signal is leaking into the inference tensor (FR-07)."
    )
    # and restore
    poisoned.df = seq.df


def test_model_input_has_exactly_six_channels_all_imu():
    """The contract input is 6 channels: accel x/y/z, gyro x/y/z. Nothing else
    can be wired in without a contract MAJOR bump."""
    from anchor.contract import FEATURE_ORDER
    assert FEATURE_ORDER == ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    assert NUM_FEATURES == 6
