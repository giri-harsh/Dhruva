"""Re-export the frozen backend-API Pydantic models from the contract stub.

`contracts/backend_api/stub_api.py` is the single source of truth for every
request/response shape and `API_CONTRACT_VERSION` (contracts/VERSIONING.md).
The real service implements logic BEHIND these models; it never redefines them.
Adding an endpoint = extend stub_api.py's models, regen openapi.json, bump
API_CONTRACT_VERSION, message Kamal — then import the new model here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_STUB = _REPO / "contracts" / "backend_api" / "stub_api.py"

_spec = importlib.util.spec_from_file_location("_anchor_api_contract", _STUB)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)  # type: ignore[union-attr]

API_CONTRACT_VERSION: str = _m.API_CONTRACT_VERSION
CamelModel = _m.CamelModel
MapRegion = _m.MapRegion
HealthResponse = _m.HealthResponse
MapExtractResponse = _m.MapExtractResponse
ModelVersionResponse = _m.ModelVersionResponse
LabelPair = _m.LabelPair
LabelUploadRequest = _m.LabelUploadRequest
LabelUploadResponse = _m.LabelUploadResponse

__all__ = [
    "API_CONTRACT_VERSION", "CamelModel", "MapRegion", "HealthResponse",
    "MapExtractResponse", "ModelVersionResponse", "LabelPair",
    "LabelUploadRequest", "LabelUploadResponse",
]
