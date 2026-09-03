"""Model registry — filesystem-backed for the hackathon (swap for object storage
+ Postgres later without touching the router).

A published model is a directory under `settings.registry_dir/<model_version>/`
containing:
    anchor_net.onnx           the exported graph (real trained weights)
    model_manifest.json       copied from contracts/model_io at export time
    manifest.sig              detached signature over sha256(anchor_net.onnx)
                              — produced in CI only (FR-25); never on a dev box
    registry.json             {modelVersion, contractVersion,
                               minSupportedContractVersion, sha256, sizeBytes,
                               publishedAt, notes}

`latest.txt` holds the current model_version. The signing PRIVATE key never
lives here — verification uses the public key pinned in `settings`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import settings


@dataclass
class RegisteredModel:
    model_version: str
    contract_version: str
    min_supported_contract_version: str
    sha256: str
    size_bytes: int
    published_at: str
    onnx_path: Path
    signed: bool
    notes: str = ""


def _dir() -> Path:
    settings.registry_dir.mkdir(parents=True, exist_ok=True)
    return settings.registry_dir


def list_models() -> list[RegisteredModel]:
    out = []
    for d in sorted(_dir().iterdir()):
        reg = d / "registry.json"
        if not reg.is_file():
            continue
        j = json.loads(reg.read_text())
        out.append(RegisteredModel(
            model_version=j["modelVersion"],
            contract_version=j["contractVersion"],
            min_supported_contract_version=j["minSupportedContractVersion"],
            sha256=j["sha256"], size_bytes=j["sizeBytes"],
            published_at=j["publishedAt"], onnx_path=d / "anchor_net.onnx",
            signed=(d / "manifest.sig").is_file(), notes=j.get("notes", ""),
        ))
    return out


def latest() -> RegisteredModel | None:
    p = _dir() / "latest.txt"
    if p.is_file():
        want = p.read_text().strip()
        for m in list_models():
            if m.model_version == want:
                return m
    models = list_models()
    return models[-1] if models else None
