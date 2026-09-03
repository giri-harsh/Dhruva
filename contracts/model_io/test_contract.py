"""
Regenerates the stub model and checks it is byte-for-byte reproducible,
and that a fresh onnxruntime inference run still matches every committed
golden vector. This is what CI runs. If someone bumps onnxruntime or onnx
and a numeric result shifts outside tolerance, this fails the build
instead of shipping a silent mismatch to the Android side.

Usage: pytest test_contract.py
(or: python test_contract.py, for a plain run without pytest)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent


def _run_case(session, vector: dict):
    arr = np.array(vector["input"], dtype=np.float32)[None, :, :]  # add batch dim back
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    outputs = session.run(
        [manifest["outputs"] and list(manifest["outputs"].keys())[0],
         list(manifest["outputs"].keys())[1]],
        {manifest["input_name"]: arr},
    )
    mean_val = float(outputs[0][0][0])
    logvar_val = float(outputs[1][0][0])
    return mean_val, logvar_val


def test_golden_vectors_reproduce():
    onnx_path = HERE / "anchor_net_stub.onnx"
    assert onnx_path.exists(), "run generate_stub_model.py first"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    golden_dir = HERE / "golden_vectors"
    vector_files = sorted(golden_dir.glob("*.json"))
    assert vector_files, "no golden vectors found"

    for vf in vector_files:
        vector = json.loads(vf.read_text())
        mean_val, logvar_val = _run_case(session, vector)
        expected = vector["expected_output"]
        tol = vector["tolerance_abs"]
        keys = list(expected.keys())
        exp_mean, exp_logvar = expected[keys[0]], expected[keys[1]]
        assert abs(mean_val - exp_mean) <= tol, (
            f"{vf.name}: mean output drifted. expected={exp_mean} got={mean_val} "
            f"(tolerance={tol}). This means the ONNX runtime/opset environment "
            f"changed numeric behavior since the vector was generated — see "
            f"TOOLCHAIN.md before assuming it's a real model bug."
        )
        assert abs(logvar_val - exp_logvar) <= tol, (
            f"{vf.name}: logvar output drifted. expected={exp_logvar} got={logvar_val} "
            f"(tolerance={tol})."
        )


def test_manifest_matches_model_metadata():
    import onnx
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    model = onnx.load(str(HERE / "anchor_net_stub.onnx"))
    embedded = {p.key: p.value for p in model.metadata_props}
    assert embedded["contract_version"] == manifest["contract_version"], (
        "The ONNX file's embedded contract_version and model_manifest.json "
        "disagree. They must be regenerated together by generate_stub_model.py "
        "— never hand-edit one without the other."
    )
    assert embedded["window_size_samples"] == str(manifest["window_size_samples"])
    assert embedded["feature_order"] == ",".join(manifest["feature_order"])


if __name__ == "__main__":
    test_golden_vectors_reproduce()
    test_manifest_matches_model_metadata()
    print("OK: model contract self-consistent and golden vectors reproduce.")
