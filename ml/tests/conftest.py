import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "ml"))

IOVNBD_ROOT = Path(os.environ.get("IOVNBD_ROOT", _REPO_ROOT / "data" / "raw" / "IO-VNBD"))
_SYNC = IOVNBD_ROOT / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset"


@pytest.fixture(scope="session")
def iovnbd_root() -> Path:
    if not _SYNC.is_dir() or not any(_SYNC.rglob("S-*.csv")):
        pytest.skip(f"IO-VNBD synchronised subset not materialised at {IOVNBD_ROOT} "
                    f"(git lfs pull the synchronised CSVs to run this test)")
    return IOVNBD_ROOT


@pytest.fixture(scope="session")
def sequences(iovnbd_root):
    from anchor.data.sync import discover_sequences
    return discover_sequences(iovnbd_root)
