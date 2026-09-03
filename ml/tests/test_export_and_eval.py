import numpy as np
import pytest
import torch

from anchor.contract import NUM_FEATURES, WINDOW_SIZE_SAMPLES
from anchor.eval.geo import integrate_speed_heading, lla_to_local_enu, rigid_align_2d
from anchor.models.anchornet import AnchorNet, AnchorNetConfig


def test_lla_enu_origin_is_zero():
    e, n = lla_to_local_enu([52.5, 52.6], [-1.5, -1.4], 52.5, -1.5)
    assert e[0] == pytest.approx(0.0, abs=1e-6)
    assert n[0] == pytest.approx(0.0, abs=1e-6)
    assert n[1] > 10_000        # 0.1 deg lat ~ 11 km
    assert e[1] > 5_000


def test_rigid_align_recovers_known_transform():
    rng = np.random.default_rng(0)
    src_e = rng.normal(size=50) * 100
    src_n = rng.normal(size=50) * 100
    th = 0.3
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = (R @ np.c_[src_e, src_n].T).T + np.array([12.0, -7.0])
    ae, an = rigid_align_2d(src_e, src_n, dst[:, 0], dst[:, 1])
    assert np.allclose(ae, dst[:, 0], atol=1e-6)
    assert np.allclose(an, dst[:, 1], atol=1e-6)


def test_onnx_export_parity(tmp_path):
    """AnchorNet -> ONNX -> onnxruntime reproduces torch within tolerance, and
    the graph has the contract's I/O names. Guards the opset-17 primitive-only
    _ChannelNorm choice (no LayerNormalization op)."""
    onnx = pytest.importorskip("onnx")
    from anchor.export.to_onnx import export_anchornet

    net = AnchorNet(AnchorNetConfig(dropout=0.0))
    ckpt = tmp_path / "m.pt"
    torch.save(net.state_dict(), ckpt)
    out = export_anchornet(ckpt, tmp_path / "anchor.onnx", model_cfg=AnchorNetConfig(dropout=0.0))

    model = onnx.load(str(out))
    names_in = [i.name for i in model.graph.input]
    names_out = [o.name for o in model.graph.output]
    assert names_in == ["imu_window"]
    assert names_out == ["velocity_mean_mps", "velocity_log_variance"]
    opsets = {op.domain: op.version for op in model.opset_import}
    assert opsets.get("", 0) <= 17
    used_ops = {n.op_type for n in model.graph.node}
    assert "LayerNormalization" not in used_ops, "must stay off the opset-17 LN op"
    # _verify_parity already asserted torch/ort agreement inside export_anchornet


def test_integrate_speed_heading_east():
    e, n = integrate_speed_heading(np.full(50, 4.0), np.full(50, np.pi / 2), 0.1)
    assert e[-1] == pytest.approx(20.0)     # 4 m/s * 5 s due east
    assert abs(n[-1]) < 1e-9
