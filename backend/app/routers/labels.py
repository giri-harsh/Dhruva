"""POST /v1/telemetry/labels — opt-in flywheel label upload (S-19, PRD §5.2).

Accepts (IMU window -> GNSS-derived displacement) pairs, validates + bounds-
checks + consent-re-checks server side, and appends accepted pairs to the
flywheel store for the next retraining pass. Never in the positioning loop.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..contracts import LabelUploadRequest, LabelUploadResponse
from ..ingestion.validate import validate_batch

router = APIRouter(tags=["telemetry"])

_FLYWHEEL = settings.registry_dir.parent / "_flywheel"


def _consent_on_file(device_id_hash: str) -> bool:
    # placeholder: a real deployment checks the consent table. Dev default: allow.
    consent_file = _FLYWHEEL / "consent.json"
    if consent_file.is_file():
        return device_id_hash in json.loads(consent_file.read_text()).get("granted", [])
    return settings.env == "dev"


@router.post("/v1/telemetry/labels", response_model=LabelUploadResponse)
def upload_labels(req: LabelUploadRequest) -> LabelUploadResponse:
    if not req.pairs:
        raise HTTPException(status_code=400, detail="pairs must be non-empty")

    consent_ok = _consent_on_file(req.device_id_hash)
    accepted, rejected, reasons = validate_batch(req.pairs, consent_ok=consent_ok)
    batch_id = f"batch-{uuid.uuid4().hex[:12]}"

    if accepted:
        _FLYWHEEL.mkdir(parents=True, exist_ok=True)
        rec = {
            "batch_id": batch_id,
            "device_id_hash": req.device_id_hash,
            "received_utc": datetime.now(timezone.utc).isoformat(),
            "pairs": [p.model_dump(by_alias=False) for p in req.pairs
                      if _keep(p)],
        }
        (_FLYWHEEL / f"{batch_id}.json").write_text(json.dumps(rec), encoding="utf-8")

    return LabelUploadResponse(
        accepted=accepted, rejected=rejected,
        rejection_reasons=sorted(set(reasons)), batch_id=batch_id,
    )


def _keep(pair) -> bool:
    from ..ingestion.validate import validate_pair
    return validate_pair(pair) is None
