"""GET /v1/model/version — serve the current signed model artefact metadata.

Tier 2b (PRD §5.1): a device PULLS this when it next has connectivity, never
inline with positioning. This endpoint owns the server side of the
compatibility-refusal rule (contracts/VERSIONING.md §2.6 / FR-24): when a model
whose contract moved to a new MAJOR is published, `minSupportedContractVersion`
on this response is set accordingly, or Kamal's app defeats its own refusal.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..contracts import ModelVersionResponse
from ..registry.store import latest

router = APIRouter(tags=["models"])


@router.get("/v1/model/version", response_model=ModelVersionResponse)
def model_version() -> ModelVersionResponse:
    m = latest()
    if m is None:
        # no real model published yet — echo the stub contract so the Android
        # client still gets a valid, refusable response
        return ModelVersionResponse(
            model_version="0.0.0-none",
            contract_version="1.0.0",
            min_supported_contract_version=settings.min_supported_model_contract,
            download_url="",
            sha256="0" * 64,
            size_bytes=0,
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    if not m.signed and settings.env != "dev":
        raise HTTPException(status_code=503, detail="latest model is unsigned; refusing to serve")
    return ModelVersionResponse(
        model_version=m.model_version,
        contract_version=m.contract_version,
        min_supported_contract_version=m.min_supported_contract_version,
        download_url=f"/v1/model/artifact/{m.model_version}/anchor_net.onnx",
        sha256=m.sha256,
        size_bytes=m.size_bytes,
        published_at=datetime.fromisoformat(m.published_at),
    )
