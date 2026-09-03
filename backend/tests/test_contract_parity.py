"""The real backend's /v1 endpoints must stay schema-identical to the frozen
`contracts/backend_api/openapi.json`. The dashboard surface is exempt.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

_REPO = Path(__file__).resolve().parents[2]
_FROZEN = json.loads((_REPO / "contracts" / "backend_api" / "openapi.json").read_text())

client = TestClient(app)


def _v1_paths(spec):
    return {p: v for p, v in spec["paths"].items() if p.startswith("/v1/")}


def test_v1_paths_match_frozen_contract():
    live = app.openapi()
    frozen_v1 = _v1_paths(_FROZEN)
    live_v1 = _v1_paths(live)
    assert set(live_v1) == set(frozen_v1), (
        f"v1 path set drifted.\n frozen: {sorted(frozen_v1)}\n live:   {sorted(live_v1)}"
    )
    for path, methods in frozen_v1.items():
        for method, op in methods.items():
            live_op = live_v1[path][method]
            assert _shape(op) == _shape(live_op), f"{method.upper()} {path} schema drift"


def _shape(op: dict):
    """Compare the parts that matter for wire compatibility: request body ref,
    parameter names/required, response 200 ref."""
    params = sorted((p["name"], p.get("required", False), p.get("in"))
                    for p in op.get("parameters", []))
    req = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
    resp = (op.get("responses", {}).get("200", {})
            .get("content", {}).get("application/json", {}).get("schema"))
    return {"params": params, "request": req, "response": resp}


def test_v1_component_schemas_match():
    frozen = _FROZEN.get("components", {}).get("schemas", {})
    live = app.openapi().get("components", {}).get("schemas", {})
    for name, schema in frozen.items():
        assert name in live, f"contract schema {name} missing from live API"
        assert schema.get("properties", {}).keys() == live[name].get("properties", {}).keys(), (
            f"schema {name} property set drifted"
        )
        assert schema.get("required", []) == live[name].get("required", []), (
            f"schema {name} required-field set drifted"
        )


def test_endpoints_respond():
    assert client.get("/v1/health").json()["status"] == "ok"
    assert client.get("/v1/model/version").status_code == 200
    assert client.get("/v1/map/extract", params={"region": "hill_corridor"}).status_code == 200
    assert client.get("/v1/map/extract", params={"region": "not_a_region"}).status_code == 422


def test_label_bounds_reject_bad_pairs():
    bad_speed = {"deviceIdHash": "d", "pairs": [{
        "imuWindow": [[0, 0, 9.8, 0, 0, 0]] * 20, "displacementM": 10_000.0,
        "windowDurationS": 2.0, "deviceModel": "x", "appVersion": "1",
        "contractVersion": "1.0.0"}]}
    r = client.post("/v1/telemetry/labels", json=bad_speed)
    assert r.status_code == 200 and r.json()["rejected"] == 1

    degps_gyro = {"deviceIdHash": "d", "pairs": [{
        "imuWindow": [[0, 0, 9.8, 200, 200, 200]] * 20, "displacementM": 20.0,
        "windowDurationS": 2.0, "deviceModel": "x", "appVersion": "1",
        "contractVersion": "1.0.0"}]}
    r = client.post("/v1/telemetry/labels", json=degps_gyro)
    assert r.json()["rejected"] == 1
