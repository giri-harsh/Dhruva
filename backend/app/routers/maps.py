"""GET /v1/map/extract — offline OSM region extract hosting (PRD §5.2).

Serves versioned + checksummed `.osm.pbf` extracts for the demo corridors.
`.pbf` files are NOT in git (.gitignore); the build scripts live in `maps/` and
the built artefacts under `settings.map_extract_dir`. `MapRegion` is a closed
enum — adding a region is a contract change (bump API_CONTRACT_VERSION).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..contracts import MapExtractResponse, MapRegion

router = APIRouter(tags=["maps"])


@router.get("/v1/map/extract", response_model=MapExtractResponse)
def map_extract(region: MapRegion) -> MapExtractResponse:
    meta_path = settings.map_extract_dir / region.value / "extract.json"
    if not meta_path.is_file():
        if settings.env == "dev":
            return MapExtractResponse(
                region=region, map_version="0.0.0-stub",
                download_url=f"/v1/map/artifact/{region.value}.osm.pbf",
                sha256="0" * 64, size_bytes=0,
                updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        raise HTTPException(status_code=404, detail=f"no extract built for region {region.value}")
    j = json.loads(meta_path.read_text())
    return MapExtractResponse(
        region=region, map_version=j["mapVersion"],
        download_url=j["downloadUrl"], sha256=j["sha256"],
        size_bytes=j["sizeBytes"], updated_at=datetime.fromisoformat(j["updatedAt"]),
    )
