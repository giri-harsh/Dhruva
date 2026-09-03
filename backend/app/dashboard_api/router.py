"""Evaluation-dashboard API (S-17, PRD §5.2). Serves the plots/tables the `web/`
frontend renders: the 13-row ablation table, the calibration reliability
diagram, the integrity ROC, the outage error-growth curves.

These are NOT part of the frozen `contracts/backend_api/` wire contract (that is
the device-facing API). The dashboard is an internal/proposal-screening
surface, so its shapes live here and can evolve freely — but everything it
serves is a committed JSON artefact from `ml/`, so every number keeps its
provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _load(name: str) -> dict:
    p = settings.dashboard_artifact_dir / name
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"{name} not generated yet")
    return json.loads(p.read_text())


@router.get("/ablation")
def ablation_table() -> dict:
    return _load("ablation_table.json")


@router.get("/calibration")
def calibration() -> dict:
    """Reliability-diagram points + ECE, from the latest window-level eval."""
    return _load("calibration_report.json")


@router.get("/integrity")
def integrity_roc() -> dict:
    return _load("integrity_roc.json")


@router.get("/baselines")
def baselines() -> dict:
    results_dir = settings.dashboard_artifact_dir.parent / "bench" / "results"
    latest = sorted(results_dir.glob("baselines_*.json"))
    if not latest:
        raise HTTPException(status_code=404, detail="no baseline results yet")
    return json.loads(latest[-1].read_text())


@router.get("/manifest")
def manifest() -> dict:
    """What artefacts exist + their provenance, for the dashboard's header."""
    d = settings.dashboard_artifact_dir
    out = {}
    for name in ("ablation_table.json", "calibration_report.json", "integrity_roc.json"):
        p = d / name
        out[name] = {"present": p.is_file(),
                     "mtime": p.stat().st_mtime if p.is_file() else None}
    return out
