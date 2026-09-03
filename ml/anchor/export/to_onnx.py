"""Export a trained AnchorNet to ONNX matching the frozen model_io contract.

The exported graph's I/O — input name `imu_window` [1,20,6], outputs
`velocity_mean_mps` / `velocity_log_variance` [1,1], opset 17, normalisation
NOT baked in — is defined by `contracts/model_io/generate_stub_model.py`. This
exporter produces a graph with the SAME interface but real trained weights.

Once weights are good (Weeks 3-5), the real per-channel NORM_MEAN / NORM_STD
(train-fit) and the exported weights replace the stub: update
`generate_stub_model.py`'s constants + point it at the trained checkpoint,
re-run it, regenerate golden vectors from the real model, run test_contract.py,
bump `modelVersion` (not `contract_version` — interface unchanged), message
Kamal (PRD §8 Week 3-5 SYNC points).

This module is the bridge used by that update: `export_anchornet(ckpt) -> Path`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..contract import (
    INPUT_NAME,
    NUM_FEATURES,
    OUTPUT_LOGVAR_NAME,
    OUTPUT_MEAN_NAME,
    WINDOW_SIZE_SAMPLES,
)
from ..models.anchornet import AnchorNet, AnchorNetConfig

OPSET = 17


class _ContractWrapper(torch.nn.Module):
    """Fixes output ORDER and names to the contract; drops aux heads from the
    exported graph (Head C/D are consumed via separate outputs only if the
    contract is bumped to expose them)."""
    def __init__(self, net: AnchorNet):
        super().__init__()
        self.net = net

    def forward(self, imu_window: torch.Tensor):
        out = self.net(imu_window)
        return out[OUTPUT_MEAN_NAME], out[OUTPUT_LOGVAR_NAME]


def export_anchornet(
    checkpoint_path: str | Path,
    out_path: str | Path,
    *,
    model_cfg: AnchorNetConfig | None = None,
) -> Path:
    net = AnchorNet(model_cfg or AnchorNetConfig())
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    wrapper = _ContractWrapper(net).eval()

    dummy = torch.zeros(1, WINDOW_SIZE_SAMPLES, NUM_FEATURES, dtype=torch.float32)
    out_path = Path(out_path)
    torch.onnx.export(
        wrapper, (dummy,), str(out_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_MEAN_NAME, OUTPUT_LOGVAR_NAME],
        opset_version=OPSET,
        dynamic_axes=None,           # batch fixed at 1, per contract
        do_constant_folding=True,
    )
    _verify_parity(net, out_path)
    return out_path


def _verify_parity(net: AnchorNet, onnx_path: Path, n: int = 16) -> None:
    """Torch vs onnxruntime must agree — the same guarantee the golden vectors
    give across languages, checked here at export time."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (n, WINDOW_SIZE_SAMPLES, NUM_FEATURES)).astype(np.float32)
    with torch.no_grad():
        t_out = net(torch.from_numpy(x))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for i in range(n):
        o = sess.run(None, {INPUT_NAME: x[i:i + 1]})
        assert np.allclose(o[0], t_out[OUTPUT_MEAN_NAME][i:i + 1].numpy(), atol=1e-4), \
            "torch/onnxruntime mean parity failed"
        assert np.allclose(o[1], t_out[OUTPUT_LOGVAR_NAME][i:i + 1].numpy(), atol=1e-4), \
            "torch/onnxruntime logvar parity failed"
