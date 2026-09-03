# `ml/` — ANCHOR-Net training + evaluation library

Training/eval library only. **No HTTP surface** — runnable from a notebook or CI.
The FastAPI service in `backend/` schedules this and serves its outputs. Keep that
split (PRD-ML-BACKEND.md §5.3).

## Layout (PRD §5.3)

```
ml/anchor/
├── contract.py     Re-exports the frozen model I/O contract from
│                    contracts/model_io/generate_stub_model.py — NEVER redefine
│                    window size / feature order / output names anywhere else.
├── data/           IO-VNBD loaders, unit conversion at the boundary, sync joiner
├── splits/         Leakage-safe split protocol as code; manifests are COMMITTED
├── models/         anchornet.py — dilated TCN trunk + heads A/B/(C/D)
├── train/          training loop, augmentation, calibration
├── eval/           metrics (§6.7), outage simulator, 13-row ablation runner
├── integrity/      GNSS attack injector + ROC bench (FR-31)
├── golden/         frozen 40-segment outage set + SHA-256 manifest
├── bench/          run_baselines.py — one command, all runnable baselines
└── export/         PyTorch→ONNX, int8 quantization, manifest signing (FR-25)
ml/tests/           unit + regression tests, incl. leakage-protocol guards
ml/docs/            IO-VNBD-verification.md and other findings
```

## Setup

Python **3.11.x only** (TOOLCHAIN.md). From the repo root:

```
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

torch is the **CPU** wheel (no GPU on either dev machine):
`uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`

## Dataset

IO-VNBD is LFS-backed (~2.2 GB). Do **not** full-clone it. Pointers-only:

```
cd data/raw
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/onyekpeu/IO-VNBD.git
```

then `git lfs pull --include="<path glob>"` per sequence as needed.
`data/raw/` is git-ignored. Findings so far: `ml/docs/IO-VNBD-verification.md`.

> ⚠ On Windows, do not run a multi-GB LFS smudge concurrently with other `git`
> commands in this repo — Defender + disk contention makes git misread its own
> object store. Pull LFS blobs as the only git activity.

## Before every push touching `contracts/model_io/`

`ml/anchor/contract.py` imports the contract; run the contract CI locally first:

```
cd contracts/model_io && python generate_stub_model.py && pytest test_contract.py -v
```

See PRD-ML-BACKEND.md §10 for the full pre-push checklist.
