"""
Stub FastAPI backend — the running, executable form of the backend API
contract. The ML/backend track fills in the real implementation behind
each endpoint over time; the Android track codes its network client
against THIS running stub from day one, using the OpenAPI spec it serves.

Run it:
    uvicorn stub_api:app --reload --port 8000

Then:
    http://localhost:8000/docs        - interactive API explorer
    http://localhost:8000/openapi.json - machine-readable spec (feed this
                                          to openapi-generator for a Kotlin
                                          client, or hand-write against it)

Wire format rule: Python stays snake_case internally; every model below
uses alias_generator=to_camel + populate_by_name=True, so JSON on the
wire is camelCase (native to Kotlin/Retrofit) without either side hand-
translating field names. This is the #1 source of silent integration
bugs in cross-language APIs — solved once, here, structurally.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

API_CONTRACT_VERSION = "1.0.0"


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------

class HealthResponse(CamelModel):
    status: str
    api_contract_version: str


# ---------------------------------------------------------------------------
# GET /v1/map/extract
# ---------------------------------------------------------------------------

class MapRegion(str, Enum):
    """Fixed, closed enum — not a free-text string. Adding a region is a
    contract change (bump API_CONTRACT_VERSION), not just a new string
    either side starts sending."""
    delhi_ncr = "delhi_ncr"
    hill_corridor = "hill_corridor"
    uk_metrics = "uk_metrics"


class MapExtractResponse(CamelModel):
    region: MapRegion
    map_version: str            # semver of the map extract, independent of API/model versions
    download_url: str
    sha256: str
    size_bytes: int
    updated_at: datetime


# ---------------------------------------------------------------------------
# GET /v1/model/version
# ---------------------------------------------------------------------------

class ModelVersionResponse(CamelModel):
    model_version: str                    # semver of the trained weights
    contract_version: str                 # semver of the I/O contract (contracts/model_io)
    min_supported_contract_version: str   # oldest contract_version this model file still matches
    download_url: str
    sha256: str
    size_bytes: int
    published_at: datetime


# ---------------------------------------------------------------------------
# POST /v1/telemetry/labels  (flywheel opt-in upload — §4.3 of the PRD)
# ---------------------------------------------------------------------------

class LabelPair(CamelModel):
    """One (IMU window -> GNSS-derived displacement) training pair.
    Position-stripped by construction: no lat/lon field exists in this
    model at all, by design, not by omission at serialization time."""
    imu_window: list[list[float]]   # [window_size][num_features], same order as contracts/model_io
    displacement_m: float
    window_duration_s: float
    device_model: str
    app_version: str
    contract_version: str           # which model_io contract this window's feature order matches


class LabelUploadRequest(CamelModel):
    device_id_hash: str             # salted hash, never a raw device identifier
    pairs: list[LabelPair]


class LabelUploadResponse(CamelModel):
    accepted: int
    rejected: int
    rejection_reasons: list[str]
    batch_id: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ANCHOR Backend API (stub)",
    version=API_CONTRACT_VERSION,
    description=(
        "Stub implementation used to freeze the wire contract before the real "
        "backend exists. Every endpoint returns deterministic canned data. "
        "Swap implementations, keep the schemas, and this contract never breaks "
        "the Android client that was written against it."
    ),
)


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", api_contract_version=API_CONTRACT_VERSION)


@app.get("/v1/map/extract", response_model=MapExtractResponse)
def map_extract(region: MapRegion) -> MapExtractResponse:
    return MapExtractResponse(
        region=region,
        map_version="0.1.0-stub",
        download_url=f"https://example-cdn.invalid/maps/{region.value}-0.1.0.osm.pbf",
        sha256="0" * 64,
        size_bytes=123_456_789,
        updated_at=datetime.now(timezone.utc),
    )


@app.get("/v1/model/version", response_model=ModelVersionResponse)
def model_version() -> ModelVersionResponse:
    return ModelVersionResponse(
        model_version="0.1.0-stub",
        contract_version="1.0.0",
        min_supported_contract_version="1.0.0",
        download_url="https://example-cdn.invalid/models/anchor_net-0.1.0-stub.onnx",
        sha256="0" * 64,
        size_bytes=45_000,
        published_at=datetime.now(timezone.utc),
    )


@app.post("/v1/telemetry/labels", response_model=LabelUploadResponse)
def upload_labels(req: LabelUploadRequest) -> LabelUploadResponse:
    if not req.pairs:
        raise HTTPException(status_code=400, detail="pairs must be non-empty")
    return LabelUploadResponse(
        accepted=len(req.pairs),
        rejected=0,
        rejection_reasons=[],
        batch_id="stub-batch-0001",
    )


if __name__ == "__main__":
    import json
    # Dump the OpenAPI spec to disk so it can be committed and diffed in PRs
    # without anyone needing to run the server first.
    from pathlib import Path
    out = Path(__file__).parent / "openapi.json"
    out.write_text(json.dumps(app.openapi(), indent=2))
    print(f"wrote {out}")
