# Project ANCHOR — repo root

Two-person split, per `ANCHOR-PRD-v3.0-FINAL.md`:

- **Harshit** — ML + backend track. PRD: `PRD-ML-BACKEND.md`
- **Kamal** — on-device engine + app track. PRD: `PRD-ANDROID-ENGINE.md`

Both PRDs are self-contained builds of the same v3.0 PRD's relevant
sections, plus a full integration/failure-mode playbook. Read your own
PRD fully before writing code; skim the other person's so you know what
they're building against.

## Start here, together, before splitting off

1. Both read `contracts/VERSIONING.md` and `contracts/units.md`.
2. Fill in `TOOLCHAIN.md`'s Android section together (JDK/Kotlin/AGP/Gradle
   versions) — five minutes, once, prevents a week of "works on my machine."
3. Repo is on GitHub: https://github.com/giri-harsh/Dhruva. `main` is
   protected (PR + 1 approval required, no force pushes, no deletions).
   Once `contracts-ci` has run at least once (triggers automatically on
   the first push to a branch), add its three jobs to the branch
   protection rule's required status checks too.
4. From here you work independently on separate branches. The `contracts/`
   folder is the only thing you both touch, and only with a heads-up
   message first (see `VERSIONING.md`).

## What's in `contracts/`

| Folder | What it freezes | Generator script (never hand-edit the output) |
|---|---|---|
| `contracts/model_io/` | The exported model's exact input/output tensor names, shapes, units, and semantics — plus golden test vectors | `generate_stub_model.py` |
| `contracts/replay_csv/` | The sensor-replay CSV column names, order, dtypes, units, missing-value convention | `make_sample_csv.py`, checked by `validate_replay_csv.py` |
| `contracts/backend_api/` | Every backend endpoint's request/response JSON shape | `stub_api.py` (run it, or just `uvicorn stub_api:app --reload`) |
| `contracts/units.md` | The one unit system used everywhere, and the gyro deg/s-vs-rad/s bug it exists to prevent | — |
| `contracts/VERSIONING.md` | How and when to bump a contract version, and the compatibility-refusal rule | — |

`.github/workflows/contracts-ci.yml` regenerates every one of these on
every push and fails the build if the committed output doesn't match —
so a contract can't silently drift from its generator.

## ML + backend track (Harshit) — local setup

Python **3.11.x only** (see `TOOLCHAIN.md`). From the repo root:

```
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

`torch` is the **CPU** wheel — neither dev machine has an NVIDIA GPU. Exact line:

```
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` is a full `==` lockfile. Track code lives in `ml/` (training/eval
library, no HTTP) and `backend/` (FastAPI service). See `ml/README.md`.
IO-VNBD dataset handling and Day-1 findings: `ml/docs/IO-VNBD-verification.md`.
