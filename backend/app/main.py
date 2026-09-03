"""ANCHOR backend — TIER 2b (PRD §5.1). FastAPI service that schedules the
`ml/` pipeline and serves its outputs. NEVER in the on-device positioning loop:
nothing here may be assumed reachable during a trip, and no endpoint requires
synchronous availability while a device is navigating.

The device-facing wire contract is frozen in `contracts/backend_api/stub_api.py`
(imported via `app.contracts`); the OpenAPI schema this app serves must stay
diff-identical to `contracts/backend_api/openapi.json` for the four v1
endpoints. The `/dashboard/*` surface is internal and evolves freely.

Run:  uvicorn backend.app.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI

from .contracts import API_CONTRACT_VERSION
from .dashboard_api.router import router as dashboard_router
from .routers.health import router as health_router
from .routers.labels import router as labels_router
from .routers.maps import router as maps_router
from .routers.models import router as models_router

app = FastAPI(
    title="ANCHOR Backend API",
    version=API_CONTRACT_VERSION,
    description="Tier 2b service. Device-facing /v1 endpoints match "
                "contracts/backend_api/. /dashboard is internal.",
)

for r in (health_router, maps_router, models_router, labels_router):
    app.include_router(r)
app.include_router(dashboard_router)


@app.get("/")
def root() -> dict:
    return {"service": "anchor-backend", "apiContractVersion": API_CONTRACT_VERSION,
            "docs": "/docs"}
