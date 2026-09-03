from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    env: str = os.environ.get("ANCHOR_ENV", "dev")
    registry_dir: Path = Path(os.environ.get("ANCHOR_MODEL_REGISTRY_DIR",
                                             _REPO / "backend" / "_registry_store"))
    map_extract_dir: Path = Path(os.environ.get("ANCHOR_MAP_EXTRACT_DIR",
                                                _REPO / "maps" / "_build"))
    dashboard_artifact_dir: Path = Path(os.environ.get(
        "ANCHOR_DASHBOARD_DIR", _REPO / "ml" / "eval"))
    signing_public_key_path: Path = Path(os.environ.get(
        "ANCHOR_SIGNING_PUBLIC_KEY_PATH", _REPO / "ml" / "export" / "keys" / "anchor_pub.pem"))
    # the model contract MAJOR the currently-published weights require
    min_supported_model_contract: str = os.environ.get(
        "ANCHOR_MIN_MODEL_CONTRACT", "1.0.0")


settings = Settings()
