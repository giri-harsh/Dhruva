# PRD — ANCHOR: ML + Backend Track (Harshit)

**Parent document:** `/home/claude/ANCHOR-PRD-v3.0-FINAL.md` ("the v3 PRD"), Project ANCHOR, SIH 2026 PS 26168. This document is a track-scoped extraction of the v3 PRD plus a full integration/failure-mode playbook and a week-by-week plan. It is written so a fresh Claude Code session, given only this file, can start building the ML pipeline and the Python backend without needing the v3 PRD or the other track's document open.

**Owner:** Harshit. **Scope:** ANCHOR-Net (the velocity + variance model), the leakage-safe training pipeline, the evaluation harness + golden set + baselines, the GNSS integrity bench, and the Python/FastAPI backend (training + flywheel retraining, model registry/OTA, telemetry ingestion, map hosting, evaluation-dashboard API).

**Companion document:** `/home/claude/PRD-ANDROID-ENGINE.md`, owned by Kamal — covers the on-device Kotlin engine (ESKF, NHC/ZUPT, alignment, GNSS quality monitor, map matching, road-manifold constraint, magnetic memory, RTS smoothing), the Android app, the edge/CLI engine, and the Kotlin-side replay harness. **Everything in that list lives in Kamal's document, not here** — if you find yourself about to design a Kalman filter state or an Android UI element, stop; it's out of scope for this track.

**The seam between the two tracks is `/home/claude/scaffold/`** — a tested, running set of contract files that fix the exact I/O between what you build and what Kamal builds. §2 of this document restates that contract precisely for your side. Read `/home/claude/scaffold/README.md` once, at the start, regardless.

---

## 0. How to read this document

This is carved from the v3 PRD (version 3.0, FINAL). Every FR, acceptance criterion, and risk quoted below is reproduced from that document, not paraphrased, except where marked `[adapted]`. Numbers marked `[VERIFY]` are unmeasured — do not put them on a slide or in a commit message as fact until measured. §§ references (e.g. "§14.2") point at the v3 PRD's own section numbers so you can cross-reference if you ever do open it.

**What is deliberately NOT in this document** (it's all in Kamal's): the 15-state ESKF and its update equations, NHC/ZUPT, the chi-square innovation gate, alignment (roll/pitch/yaw estimation), the road-manifold constraint's filter-side mechanics, magnetic-memory correlation, OSM map matching, RTS smoothing, the Android app UI, the edge/CLI engine's Kotlin implementation, and the Kotlin side of the replay harness. You own the *model* that feeds the filter and the *server* the app occasionally talks to; you do not own anything that runs inside the positioning loop on-device.

---

## 1. Shared context (condensed from v3 PRD Part II §§1-7)

### 1.1 The problem, in one paragraph

A phone loses GNSS in tunnels, multi-level parking, and urban canyons. Existing nav apps freeze the marker, then teleport it on reacquisition. Naive IMU dead-reckoning fails because double-integrating accelerometer noise for position produces error that grows with the *square* of time — a few metres in 10 s, tens of metres in 60 s, hundreds in 180 s (v3 PRD §1.2). Wheel-tick/CAN-based INS units solve this properly but need a hardware connection no phone has and no two-wheeler carries. ANCHOR's bet: train a model to recognize vehicle speed from IMU vibration texture (a *perception* problem, non-accumulating) rather than integrate acceleration (an *integration* problem, accumulating), and fuse that speed estimate — with a **calibrated per-window uncertainty** — into a filter that also understands road geometry, so a tunnel becomes a line rather than a plane.

### 1.2 The one-sentence thesis (v3 PRD §3.1)

> "ANCHOR teaches a phone to feel how fast a vehicle is moving, and to know how sure it is — so that when the satellites disappear in a tunnel, the map keeps moving correctly instead of freezing, and says so when it cannot."

### 1.3 Why this is a substitution, not a filter-tuning problem (v3 PRD §3.2)

A Kalman filter combines information optimally; it cannot manufacture information that isn't there. Double-integrated phone acceleration contains almost no usable distance information after ~20 seconds. The fix is not a better filter — it's replacing the *source* of the velocity measurement with a learned regression whose error doesn't compound, because each window's estimate is independent. Your model is that regression. **This is the entire reason your track exists**: without a calibrated velocity+variance estimate, Kamal's filter degrades to B3 (ESKF+NHC+ZUPT, no learning) — which is a legitimate fallback (§20.1's Week-5 pivot), but the thesis of the whole project rides on your model beating it.

### 1.4 Personas (v3 PRD §7.1, verbatim, condensed)

- **P1 — Ravi**, 24, delivery rider, mid-range 4 GB Android, degraded battery, prepaid/throttled data, limited English. Needs the model to run fast on cheap hardware and to work without connectivity.
- **P2 — Sunita**, 41, ambulance driver, hill roads, long dead zones, needs the confidence signal more than anyone — a known-bad position beats an unknown-bad one.
- **P3 — Arun**, 35, fleet ops manager, buys on evidence — trip-level drift statistics, audit trail. **He is the primary consumer of your evaluation dashboard and the fleet API.**

### 1.5 The moat, and where your work sits in it (v3 PRD §4.4)

| Component | Clonable in a week? | Verdict |
|---|---|---|
| ESKF + NHC + ZUPT | Yes | No moat (Kamal's) |
| HMM map matching | Yes | No moat (Kamal's) |
| The velocity head *concept* | Yes — published by AVNet/DMDVDR (2025) and arXiv:2505.18490 | No moat — don't claim novelty for the idea itself |
| **The velocity head trained with a leakage-safe protocol** | **No — 4 to 6 weeks** | **Your moat. §3 below is the whole reason.** |
| **Calibrated uncertainty feeding the filter** | **No — the part everyone skips** | **Your moat. FR-08.** |
| **Data + magnetic flywheel on real Indian driving** | **No — needs users, not skill** | **The only moat that grows. Your backend builds the plumbing for it (§5).** |

You are building the two hardest-to-clone pieces of the entire system. Treat the leakage-safety and calibration work with the seriousness that implies — a team that random-splits gets an impressive number that collapses on a held-out driver and won't know why (§3.3 below).

---

## 2. The integration contract, restated for your side

Everything in this section is sourced from `/home/claude/scaffold/`, read directly from the files, not summarized from memory. **Treat every path, field name and value below as exact** — if you ever see a discrepancy between this section and the actual scaffold file, the file wins; re-read it.

### 2.1 What you own in `contracts/`

You own `contracts/model_io/` and `contracts/backend_api/` — you write the generators, Kamal's side only ever reads their generated output. You do **not** own `contracts/replay_csv/` (Kamal's side generates and consumes it; you may need to read `sample_replay.csv` and its schema if you build IO-VNBD→replay-CSV conversion tooling, but you don't own the schema).

### 2.2 `contracts/model_io/` — the model I/O contract

**Generator:** `contracts/model_io/generate_stub_model.py`. **Never hand-edit `anchor_net_stub.onnx`, `model_manifest.json`, or anything under `golden_vectors/` — always change the constants at the top of the generator script and re-run it.** This is not a style preference; `contracts-ci.yml` (§2.5 below) fails the build if the committed artifacts don't match a fresh regeneration.

The contract, exactly as it exists today (`contract_version` **1.0.0**):

- **Input tensor:** name `imu_window`, shape `[1, 20, 6]` (`[batch, time, features]`), dtype float32.
- **Window:** 20 samples = 2.0 s at 10 Hz (`WINDOW_SIZE_SAMPLES = 20`, `SAMPLE_RATE_HZ = 10`).
- **Feature order — fixed, positional, nothing inferred from column names downstream:**
  ```
  accel_x, accel_y, accel_z,   # m/s², vehicle-frame-aligned
  gyro_x,  gyro_y,  gyro_z,    # rad/s, vehicle-frame-aligned
  ```
  **Vehicle-frame-aligned** means Kamal's alignment stage (roll/pitch from gravity, yaw from motion — v3 PRD FR-04/FR-05) has already rotated the raw phone-frame samples into the vehicle frame *before* they reach your model. **Your model never sees phone-frame data.** This is a hard boundary: you must never build or evaluate the model against un-rotated IO-VNBD phone-frame channels without applying the same rotation convention Kamal's `AlignmentService` uses (§6.3 below — this is one of the two open decisions you must lock down together in week 1).
- **Outputs**, two heads, each `[1, 1]` float32:
  - `velocity_mean_mps` — unit m/s, "predicted forward speed for this window."
  - `velocity_log_variance` — unit `log((m/s)^2)`, **natural-log of variance, not the variance and not the std-dev.** The consumer (Kamal's `FusionService`) computes `variance = exp(velocity_log_variance)`. **You must never emit variance or std-dev directly from any model you export** — the manifest's `compatibility_rule` and the metadata embedded in the ONNX file both assert this contract, and Kamal's code is written to trust it literally.
- **Normalization:** `model_manifest.json`'s `normalization` block states `"applied_by": "caller, before inference"` — **normalization is NOT baked into the ONNX graph.** The stub's `mean`/`std` arrays are currently `[0,0,0,0,0,0]` / `[1,1,1,1,1,1]` (i.e. a no-op) because there's no real training run yet. **When you train the real model, the real per-channel mean/std (fit on the training split only, per §3.4 below) become the new `NORM_MEAN`/`NORM_STD` constants in `generate_stub_model.py`, and the manifest's `normalization.mean`/`normalization.std` arrays are what Kamal's Kotlin code must apply — bit-for-bit the same numbers — before calling `session.run()`.** There is exactly one place these numbers are allowed to be written down: this generator script. If you ever find yourself typing a normalization constant into a paper, a slide, or a message to Kamal instead of pointing at the manifest, stop — that's the exact drift failure mode §7.2 below exists to prevent.
- **Opset:** pinned to **17** (`onnx.helper.make_opsetid("", 17)`). Any architecture change that needs a newer opset (e.g. `LayerNormalization`, a custom RNN op, int8 QDQ ops) is a `contracts/model_io` **MAJOR** version bump per `VERSIONING.md`, because it changes which `onnxruntime-android` version Kamal must ship — message him before you write that code, not after (§2.6).
- **Embedded metadata:** the same contract fields (`contract_version`, `window_size_samples`, `sample_rate_hz`, `feature_order`, output names, output semantics, `norm_mean`, `norm_std`) are written into the ONNX file's own `metadata_props`, **not only into `model_manifest.json`** — so Kamal's Android code can sanity-check the file it just downloaded against what it expects at load time, without a second network call. `test_contract.py`'s `test_manifest_matches_model_metadata` asserts these two copies never disagree; if you ever regenerate one without the other, this test is what catches it.

**⚠ Open naming/semantics tension you must resolve before real training (flag this, don't silently fix it — see §9's open-decisions list).** `model_manifest.json` names the outputs `velocity_mean_mps` (unit m/s, "predicted forward speed") — but v3 PRD §14.1 specifies the model target as **scalar forward *displacement* over the window, in metres**, not instantaneous speed, precisely because "predicting speed and then integrating it re-introduces exactly the integration we are trying to eliminate." These are not the same quantity, and the current scaffold's naming says speed. Before you write the real training loop, decide with Kamal which the wire contract will actually carry — displacement in m over the 2.0 s window (matching §14.1's stated rationale, converted to an implied mean speed of `displacement / 0.2s`... no — `displacement_m / window_duration_s`) or true mean velocity in m/s (matching the current field name) — and if displacement is correct, **rename the output field and bump `contract_version` to `2.0.0`, with a message to Kamal before you push.** Do not let the field stay named `velocity_mean_mps` while training it as a displacement target, or vice versa — that's a silent semantics bug that will not throw an exception, it will just make the filter's fused position subtly and consistently wrong (see the "mean-vs-log-variance-vs-std confusion" row in §7.2).

### 2.3 `contracts/model_io/golden_vectors/` — the anti-drift mechanism

Three fixed input→output pairs: `stationary.json` (all-zero window), `accelerating_straight.json` (ramping accel_x + constant yaw), `random_vibration.json` (seeded random noise, seed 7). Each carries `input` (the `[20][6]` window with the batch dim dropped for readability), `expected_output` (both head values), and `tolerance_abs` (currently `1e-4`).

**This is the single most important anti-drift mechanism in the whole scaffold, and it is your responsibility to keep it true every time you touch the model.** The claim these vectors exist to prove: *the exact same `.onnx` file produces the exact same numbers on ONNX Runtime Mobile (Android) as it does on `onnxruntime` (Python, your training/eval environment).* Two runtimes, two languages, same weights, same file — if the numbers match within tolerance, cross-platform numeric drift is not silently happening. When you retrain with real weights, `generate_stub_model.py`'s `main()` regenerates all three golden vectors from the newly-trained model's actual output — you don't hand-write expected values, you let the script compute them from a real inference run, exactly the same way it does for the stub today.

**You run `test_contract.py` before every push that touches the model.** It does two things: `test_golden_vectors_reproduce` re-runs each golden vector through the current `.onnx` file and asserts the output hasn't drifted past `tolerance_abs` (catches an `onnxruntime`/`onnx` version bump silently changing floating-point rounding in some op); `test_manifest_matches_model_metadata` asserts the manifest and the ONNX file's own embedded metadata agree. **Kamal's side runs the Android-equivalent of the first test** (currently stubbed as a comment in `contracts-ci.yml` — see §7.7, it's his to-build item, not yours, but you should know it exists and coordinate on tolerance).

### 2.4 `contracts/backend_api/` — the wire contract for everything the app talks to over a network

**Generator:** `contracts/backend_api/stub_api.py`, a real running FastAPI app (`uvicorn stub_api:app --reload --port 8000`). `openapi.json` is generated from it (`python stub_api.py` dumps `app.openapi()` to disk) — **never hand-edit `openapi.json`.** `API_CONTRACT_VERSION = "1.0.0"`.

**Every endpoint that exists today, exactly:**

| Method | Path | Request | Success response |
|---|---|---|---|
| `GET` | `/v1/health` | — | `{status, apiContractVersion}` |
| `GET` | `/v1/map/extract?region=<enum>` | query param `region` ∈ `{delhi_ncr, hill_corridor, uk_metrics}` (closed enum — `MapRegion`) | `{region, mapVersion, downloadUrl, sha256, sizeBytes, updatedAt}` |
| `GET` | `/v1/model/version` | — | `{modelVersion, contractVersion, minSupportedContractVersion, downloadUrl, sha256, sizeBytes, publishedAt}` |
| `POST` | `/v1/telemetry/labels` | `{deviceIdHash, pairs: [{imuWindow, displacementM, windowDurationS, deviceModel, appVersion, contractVersion}]}` | `{accepted, rejected, rejectionReasons, batchId}` |

**Note these are the base path `/v1/...`, distinct from the v3 PRD's own §13 API contract (`/maps/regions`, `/devices/enrol`, `/labels/batch`, `/fleet/trips`, etc.) — the v3 PRD's §13 was written before this scaffold and describes an intended *fuller* API surface. The scaffold's four endpoints above are what actually exists and runs today, and what you build the real backend against; treat §13 of the v3 PRD as a roadmap for endpoints you'll likely add later (fleet queries, magnetic-signature upload, device enrolment), not as the current contract.** When you add one of those, you add it the same way as everything else here: extend `stub_api.py`'s Pydantic models, regenerate `openapi.json`, bump `API_CONTRACT_VERSION` per the rule in §2.6, tell Kamal before writing the Kotlin client work depends on him having received.

**The `CamelModel` pattern — why it exists, use it for every new model you add.** Every Pydantic model in `stub_api.py` inherits `CamelModel`, which sets `model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)`. This means: your Python code stays idiomatic snake_case internally (`device_id_hash`, `download_url`), but the JSON that actually goes over the wire is camelCase (`deviceIdHash`, `downloadUrl`) — which is what Kotlin/Retrofit/kotlinx.serialization expects natively on the Android side, with zero hand-translation on either end. **This is the structural fix for the single most common cross-language API bug** (a Kotlin data class expecting `downloadUrl` deserializing a Python API's `download_url` into a null field, silently, with no exception — see §7.4). Every new request/response model you write **must** subclass `CamelModel`, not plain `BaseModel`. If you ever see a raw `BaseModel` in a new endpoint you're adding, that's a bug — fix it before you push.

**`MapRegion` is a closed enum, not a free string** — `delhi_ncr`, `hill_corridor`, `uk_metrics`. Adding a region is a contract change (bump `API_CONTRACT_VERSION`), not a new string either side just starts sending — this matters because a typo'd region string on either side currently fails loudly (Pydantic validation error / Kotlin enum parse failure) rather than silently serving nothing.

### 2.5 `contracts-ci.yml` — what runs on every push, and what you're accountable for

Three of the four CI jobs are yours (`.github/workflows/contracts-ci.yml`, jobs run on `windows-latest` to match both your dev machines):

- **`model-io-contract`**: `pip install -r requirements.txt pytest`, `python generate_stub_model.py`, then `git diff --exit-code` — **fails if regenerating produces an uncommitted diff**, i.e. if you changed the generator's constants but forgot to re-run it and commit the output, or vice versa. Then `pytest test_contract.py -v`.
- **`replay-csv-contract`**: not yours to fix, but you may trigger it if you write IO-VNBD→CSV export tooling that touches `make_sample_csv.py`. Fails on the same stale-artifact pattern.
- **`backend-api-contract`**: `python stub_api.py` (regenerates `openapi.json`), then `git diff --exit-code -- openapi.json` — fails if you changed `stub_api.py`'s models without regenerating and committing `openapi.json`. The error message names the exact remedy: "Bump `API_CONTRACT_VERSION` per `VERSIONING.md`, message the other track, then commit."
- **`android-contract-check`** is commented out in the workflow file today — it's the golden-vector instrumented test on the Android side, and it's explicitly **Kamal's** to-build item (§7.7 flags this so it doesn't fall through the cracks between the two documents).

**Set branch protection so all three of your jobs are required status checks before merge into `main`** (§8's week-1 checklist has this as a literal task).

### 2.6 `contracts/VERSIONING.md` — the rule, restated for what you specifically must do

Three independent version numbers exist and you touch two of them directly:

- **`model_io` contract version** (in `model_manifest.json` and the ONNX metadata) — bump when window size, feature order, feature units, or output semantics change. **Not** bumped when you retrain with new weights but an unchanged interface.
- **`backend_api` contract version** (`API_CONTRACT_VERSION` in `stub_api.py`, reflected in `openapi.json`) — bump when any endpoint's request/response shape changes, or an endpoint is added/removed.
- **Model weights version** (the `GET /v1/model/version` response's `modelVersion` field) — **bump this every single time you publish retrained weights**, even when `contract_version` hasn't changed. This is the field that changes constantly during the flywheel; `contract_version` should change rarely.

**MAJOR / MINOR / PATCH, applied to your two contracts specifically:**
- **MAJOR** (breaking, would silently break Kamal if he didn't know): window size change, feature reorder, a feature's unit changing, an output's meaning changing (e.g. resolving the displacement-vs-velocity tension in §2.2), a required API field renamed/removed/reshaped, an API field's unit changing.
- **MINOR** (backward-compatible addition): a new optional API field, a new enum value old Kotlin code can ignore (e.g. a new `MapRegion`), a new model output head old code doesn't read.
- **PATCH**: no shape change at all — e.g. fixing a description string, correcting a comment.

**The rule that actually prevents disasters: message Kamal *before* you write code against a changed contract, not after, every single time — even for something that feels obviously safe.** For a two-person team this is a WhatsApp/Slack message, not a formal RFC, but it happens before the PR, every time, no exceptions for "it's just a rename."

**The compatibility-refusal rule is his enforcement, but you own the server-side half of it.** `GET /v1/model/version` returns `minSupportedContractVersion` — Kamal's app compares this against its own `MIN_SUPPORTED_CONTRACT_VERSION` constant and refuses to load a model outside that range, falling back per FR-24. **Your job:** when you publish a model whose `contract_version` moved to a new MAJOR, you set `minSupportedContractVersion` on the corresponding `/v1/model/version` response correctly — get this wrong (e.g. leave it at `1.0.0` when you shipped a `2.0.0`-contract model) and you defeat the entire refusal mechanism, silently, and Kamal's app will try to run a model it cannot correctly interpret.

### 2.7 What you own vs. what you must never touch

**You own, and are the only one who writes to:** `contracts/model_io/generate_stub_model.py` (and everything it generates), `contracts/backend_api/stub_api.py` (and `openapi.json`), `ml/` (the whole training/eval library), `backend/` (the FastAPI service), `reference/` (the Python oracle that generates golden vectors for the *filter*, not the model — this is Kamal's oracle to consume but yours to maintain, see v3 PRD §10.5's repo tree).

**You never touch:** `contracts/replay_csv/schema.json` or its generator (Kamal's), `core/` (the Kotlin engine), `android/` (the app), `edge/` (the CLI). If you need a change to the replay CSV schema — e.g. you want IO-VNBD conversion tooling to emit a new column — that's a message to Kamal and a `VERSIONING.md`-governed change to *his* file, not something you edit directly.

---

## 3. Your scope, in MoSCoW terms (v3 PRD §6.1, filtered to your items)

| ID | Item | Priority | In demo? | Note |
|---|---|---|---|---|
| S-01 | ANCHOR-Net velocity + variance heads, leakage-safe split | **Must** | Yes | The thesis. Yours. |
| S-08 | Evaluation harness + golden set + baseline table | **Must** | Yes | The credibility artefact. Yours. |
| S-11 | GNSS integrity bench | **Must** | One slide | Two days' work, per §5.4. Yours (ML/eval side). |
| S-15 | Motion-context head (idle/normal/rough/impulse/handling) → filter noise | **Should** | Yes if ready | Falls back to fixed `R` and Kamal's deterministic detectors if cut. Yours to train; Kamal consumes the output. |
| S-16 | Head D — learned yaw-increment correction | **Should** | No | Gated on the Week-5 ablation; cut first under pressure. Model is yours; the attitude update that consumes it is Kamal's (`AttitudeUpdate.kt`). |
| S-17 | Web evaluation dashboard | **Should** | Yes | Also the proposal-screening artefact. Its API is yours (`backend/app/dashboard_api/`); the frontend (`web/`) is a thin client on top — you may need to build both if no one else picks up the frontend. |
| S-19 | On-device label collection for the flywheel (opt-in) | **Could** | No | The upload endpoint (`POST /v1/telemetry/labels`) is yours and already stubbed; the on-device collection and opt-in UI is Kamal's. |
| S-28 | Learned GNSS-trust model | **Won't** | n/a | Explicitly replaced by Kamal's chi-square gate — **do not build this even if it seems tempting**; it was deliberately removed to keep ML scope to one model. |

**Everything else in the v3 PRD's §6.1 table (S-02 through S-07, S-09, S-10, S-12, S-13, S-14, S-18, S-20 through S-27) belongs to Kamal's track or is explicitly out of scope for this hackathon round.** If your work seems to require touching one of those, that's a sign you're drifting into his scope — check with him first.

---

## 4. Functional requirements you own (v3 PRD §8, full Given/When/Then carried over)

### The model

| ID | Requirement | Acceptance criteria |
|---|---|---|
| **FR-07** | Predict forward displacement per window from phone IMU alone. | **Given** a 2 s window of aligned IMU data, **when** ANCHOR-Net infers, **then** the velocity head returns a mean displacement and a variance, and **the inference path reads no GNSS and no wheel-speed input** — asserted by an input-provenance test that fails the build if a GNSS or CAN column reaches the model tensor. |
| **FR-08** | The predicted variance must be **calibrated**. | **Given** the golden set, **when** predictions are binned by predicted variance, **then** empirical error matches the predicted distribution with expected calibration error below `[VERIFY: target, e.g. 0.05]`, and a reliability diagram is produced. |

### Model versioning and fallback (shared ownership — you own the server/signing side)

| ID | Requirement | Acceptance criteria | Your part |
|---|---|---|---|
| **FR-24** | Fall back safely when the model is unavailable or implausible. | **Given** a missing, corrupt or out-of-bounds model output, **when** inference is attempted, **then** the engine logs it, degrades to the NHC-only filter (B3 behaviour), sets a degraded mode pill, and does not crash. | You supply the physical output bounds (§4.1's "displacement > 200 m in 2 s is impossible" style checks, from v3 PRD §14.8/§12.2) that Kamal's `ModelFallback.kt` validates against. Document these bounds explicitly in `model_manifest.json` or an adjacent contract file as you finalize them — don't leave Kamal guessing what "implausible" means numerically. |
| **FR-25** | Version and pin every model artefact. | **Given** a running engine, **when** a pose is emitted or a trace exported, **then** it carries the model version hash; loading a model whose hash is not in the signed manifest is refused. | `ml/export/sign_manifest.py` is yours: every exported `.onnx` gets SHA-256 hashed and signed; the public key is pinned in the app (Kamal's half); **the private signing key never touches a developer machine — it lives in the CI secret store** (v3 PRD §15.3). |

### Head D (shared ownership — model is yours, filter consumption is Kamal's)

| ID | Requirement | Acceptance criteria | Your part |
|---|---|---|---|
| **FR-32** | Learned yaw-increment correction (Head D). *Should.* | **Given** Head D is enabled, **when** it infers, **then** it returns a per-window yaw increment and variance fused as an attitude measurement; **and** the Week-5 ablation shows a heading-error improvement over Head-D-disabled at 5 seeds, otherwise the head is cut and the PRD records that it was. | You train Head D and run the ablation (`ml/tests/test_head_d_ablation_gate.py`, per the v3 PRD's traceability matrix). You report the Week-5 result to the team; if it doesn't clear the bar, you cut it from the exported model *before* Kamal builds `AttitudeUpdate.kt` around a dead head — tell him the decision the day the gate is evaluated, not after he's built against it. |

### GNSS integrity bench (yours, ML/eval)

| ID | Requirement | Acceptance criteria |
|---|---|---|
| **FR-31** | GNSS integrity bench. | **Given** a held-out sequence and an attack specification (`step`, `drag`, `jam`, `multipath`) with a swept parameter, **when** the bench runs, **then** it emits a detection ROC (detection rate vs false-rejection rate on clean fixes) and a summary naming the detection threshold and **the regime that is provably undetected**. Reproducible from a fixed seed and checked against a committed expected curve. |

Note: FR-31 tests the *detection gate itself*, which is Kamal's `ChiSquareGate.kt`. Your job is the attack injector and ROC scoring (`ml/integrity/attacks.py`, `ml/integrity/roc.py`) — you inject synthetic attacks into replay sequences and score how Kamal's gate responds; you do not write the gate.

> **Note on §8's own framing (v3 PRD):** "Several of them are requirements that the system *refuse* to do something." Your two refusal-relevant requirements are FR-07's provenance assertion (refuse to let GNSS/CAN data reach the model tensor — this is a build-time test you own) and FR-24's bounds (refuse to trust an implausible output — you own the bounds, Kamal owns the enforcement).

---

## 5. Architecture — Tier 2b, and where your code lives (v3 PRD §10.0, §10.2b, §10.5)

### 5.1 The three-tier statement, and your place in it

The v3 PRD splits the middle "Application" tier into **2a** (Kamal's on-device engine, Kotlin/JVM, must run fully offline) and **2b** (your Python backend, FastAPI, assumes connectivity). **The relationship between them is one-directional and asynchronous**: your backend trains a model over hours or days from accumulated opt-in data, signs it, and a device *pulls* it whenever it next has connectivity — never the reverse, and never inline with a positioning update. **FR-22 ("zero network access for the entire trip") is a constraint on Kamal's tier, not yours** — but it means your backend must never be assumed reachable by anything in the positioning loop, and no requirement of your API contract may implicitly require synchronous availability during a trip.

### 5.2 Your five responsibilities, exactly as v3 PRD §10.2b names them

| Responsibility | What it does | Repo location |
|---|---|---|
| **Model training & flywheel retraining** | PyTorch training loop on committed split manifests; ingests opt-in label pairs from `/v1/telemetry/labels`; retrains on the growing distribution; re-runs the golden-set gate before any retrained model ships | `ml/train/`, scheduled/exposed by `backend/app/` |
| **Offline map-extract hosting and distribution** | Builds and serves `.osm.pbf` region extracts; versions and checksums them | `maps/` (build scripts, no `.pbf` in git — see `.gitignore`), served via `GET /v1/map/extract` |
| **Model registry and OTA model-push** | Signs each exported `.onnx`, serves `GET /v1/model/version`, enforces min-contract-version gating | `ml/export/sign_manifest.py`, `backend/app/registry/` |
| **Evaluation / replay web dashboard's API** | Serves plots, ablation table, calibration reliability diagram, integrity ROC | `backend/app/dashboard_api/`, rendered by `web/` |
| **Telemetry ingestion** | `/v1/telemetry/labels` (and, per §13's fuller roadmap, later `/labels/batch`-style endpoints, magnetic-signature upload, fleet queries) — validated, bounds-checked, consent-checked | `backend/app/ingestion/` |

**What you explicitly never do:** compute a live position, see a trip in progress, or sit in the critical path of the on-device latency budget (v3 PRD §9.1). If a design of yours would require the app to make a network call during positioning, that design is wrong — stop and redesign, don't ship it and hope for connectivity.

### 5.3 Repo layout you own (v3 PRD §10.5, filtered)

```
ml/                        Training + evaluation. Never shipped to a device.
├── data/                  IO-VNBD loaders, schema validation, synchronised joiner
├── splits/                §I.4 protocol as code. Manifests are COMMITTED artefacts.
├── models/                anchornet.py — shared trunk + heads
├── train/                 Training loop, augmentation, calibration
├── eval/                  Metrics (§14.6), outage simulator, ablation runner
├── integrity/             Attack injector + ROC bench (FR-31)
├── golden/                Frozen 40-segment set + SHA-256 manifest
├── bench/run_baselines.py One command, all runnable baselines, versioned JSON
└── export/                PyTorch -> ONNX, quantisation, manifest signing

reference/                 Python reference — the ORACLE for the FILTER, not the model.
├── anchor_ref/            NumPy mirror of Kamal's core/. Readable over fast.
└── golden/                Generates the golden vectors core/ must reproduce
                            (you maintain this so Kamal's filter has something
                            to regression-test against; you do not implement
                            the filter itself)

backend/                   TIER 2b — Python/FastAPI. Never in the positioning loop.
├── app/
│   ├── main.py             FastAPI app, OpenAPI 3.1 spec is the source of truth
│   ├── routers/            maps.py · models.py · labels.py (extend as §13 grows)
│   ├── registry/           Model signing, versioning, min-contract-version gating
│   ├── ingestion/          Label upload — bounds + consent checks
│   ├── dashboard_api/      Serves web/'s plots, ablations, calibration, ROC
│   └── db/                 Postgres models: trip/fleet/consent metadata
└── tests/                  Contract tests generated from openapi.json

web/                        Evaluation dashboard frontend (thin client on backend/)
maps/                       OSM extract build scripts + checksums. No .pbf in git.
```

**Note on `ml/` vs `backend/` (v3 PRD §10.5):** `ml/` is the training/eval *library* — no HTTP surface, runnable from a notebook or CI job. `backend/` is the thin FastAPI *service* that schedules `ml/`'s pipeline and serves its outputs over HTTP. Keep this split real: don't let `backend/app/` accumulate ML logic that belongs in `ml/`, and don't let `ml/` grow an HTTP dependency.

---

## 6. Dataset, leakage protocol, and model design (v3 PRD Part I + §14, full)

*This section is the core of your technical work. It is carried over in full because it's the part a technical judge reads carefully, and because getting it wrong invalidates every headline number the project reports.*

### 6.1 IO-VNBD — what's actually in it (v3 PRD §I.1–I.3)

**Source:** Onyekpe, Palade, Kanarachos, Szkolnik, *IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning*, arXiv:2005.01701 / *Data in Brief*, 2021.

- **Vehicle stream (`V-*`)**: CAN bus + Racelogic VBOX, 10 Hz — GPS, four wheel speeds (FL/FR/RL/RR, rad/s), yaw rate, vehicle speed, steering angle, longitudinal/lateral acceleration, handbrake, gear, engine rpm, and more. ~40 h, ~1,300 km, England.
- **Smartphone stream (`S-*`)**: Android, IMU 10 Hz, GPS 1 Hz `[VERIFY from Tables 3–4]` — GPS, accelerometer XYZ, gravity XYZ, gyroscope yaw/pitch/roll, magnetometer XYZ, device orientation. ~58 h, ~4,400 km, England/France/Nigeria.
- **Ground truth:** VBOX GPS. Metre-class, not centimetre-class — every error figure is bounded below by this.

**The single most important thing in this dataset:** the repository separates **"Synchronised V and S"** from **"Unsynchronised"**. In synchronised drives, the phone and the CAN bus recorded at the same moment. That gives you `(phone IMU → wheel-speed-derived displacement)` pairs — train the model to predict what the wheel sensors said, from the phone alone, then discard the wheel sensors at inference. **`[VERIFY — Day 1, hour 1]` the size of the synchronised subfolder is not stated in the paper abstract and gates the entire supervised training plan.** This is literally the first thing you measure, before writing any model code.

**What the dataset does *not* contain, stated openly rather than designed around:** no tunnels/parking sequences (you synthesize GNSS outages by masking, 30/60/120/180 s, matching WhONet's protocol), no two-wheelers, only England/France/Nigeria (no India — domain-shift risk R-04), no phone-remount labels (synthesized via SO(3) augmentation), nothing above 10 Hz IMU (your high-rate pre-filtering claims are validated on your own captures, not IO-VNBD — say so explicitly wherever you report them), no audio or barometric data.

### 6.2 The leakage trap — four ways to fool yourself (v3 PRD §I.4, full)

A naive random split of this dataset **is guaranteed to leak**, in four ways:

1. **Window overlap.** A 2 s window at 10 Hz with 1-sample stride shares 19 of 20 samples with its neighbor. Random split puts near-identical windows on both sides of train/test.
2. **Intra-drive correlation.** Windows 30 s apart in the same drive share road surface, weather, tyre pressure, driver behaviour, mounting angle — not independent samples.
3. **Driver/vehicle/phone identity.** A model that memorizes "this is the Moto G7 in this car with this driver" scores well and generalizes to nothing.
4. **Route repetition.** Sequence families (`V-Vta*`, `V-Vtb*`, `V-Vw*`) are repeated runs of the same road on different days. Random splitting puts the same geometry on both sides.

**Split protocol — held out at whole-sequence level, never window level:**

| Split | Content | Purpose |
|---|---|---|
| Train | Synchronised England sequences, drivers A–D | Fit ANCHOR-Net |
| Validation | Two whole England sequences from a driver not in train | Early stopping, hyperparameters |
| Test (in-distribution) | Whole England sequences, unseen driver + unseen route | Headline numbers. Touched at most twice. |
| Test (out-of-distribution) | France (`S-T*`) and Nigeria (`S-I`) | Generalization, reported separately and honestly |
| Repeat-route pairs | Matched pairs from `V-Vta*`/`V-Vtb*`/`V-Vw*`, both held out | Evaluates magnetic route memory (FR-30) — Kamal's, but you supply the split |
| Golden set | 40 frozen outage segments from test splits, checksummed, committed | CI regression gate. Never used for tuning. |

**Three hygiene rules, enforced in code and tested:** no window crosses a sequence boundary (`SequenceWindower`, `test_no_cross_sequence_windows`); normalisation statistics fitted on train only, serialized with the model (`test_normaliser_fitted_on_train_only`); a 10-second guard band dropped at every split boundary.

**Split manifests are committed files, not code that regenerates them** — `ml/splits/*.json` — so any reported number traces to an exact, immutable list of sequences.

### 6.3 Baselines you must implement (v3 PRD §I.5)

| # | Baseline | What it has | Note |
|---|---|---|---|
| B1 | Constant-velocity extrapolation | Last GNSS velocity, held | The honest zero-line |
| B2 | Strapdown INS, no learning | Phone accel+gyro, double integration | The physics-only path — Kamal implements this in Python `reference/`; you consume the result |
| B3 | ESKF + NHC + ZUPT, no learned velocity | Phone IMU + kinematic constraints | **The ablation that decides whether the ML earns its place.** Kamal's filter, your ablation harness runs it. |
| B4 | WhONet — cited, not run | Wheel-speed sensors | Reports up to 93% error reduction after 180 s. Cited only. |
| B5 | AVNet/DMDVDR — cited, not run | Smartphone IMU only | 0.64% drift after 578 m loss. **This is your real bar, not the PS's 10% threshold.** |

**Fair comparison requires:** identical outage segments, identical ground truth, identical metric definitions (§6.6 below), every runnable baseline executed by the *same* harness — `ml/bench/run_baselines.py`, one command, versioned JSON output.

### 6.4 Input representation, windowing, and the vehicle-frame boundary (v3 PRD §14.1)

**Rate split:** IO-VNBD's phone IMU is 10 Hz; a real phone delivers 100–200 Hz. Hampel despiking, adaptive notch, anti-alias low-pass run at device-native rate (Kamal's, **not validated by IO-VNBD** — say so explicitly wherever this is reported); decimation to 10 Hz; **your model runs at 10 Hz, the dataset's native rate, and is the one stage fully validated by IO-VNBD.**

**Frame:** every input is rotated from phone body frame to **vehicle frame** by Kamal's alignment estimate before it reaches your model. Removing this nuisance variable is why the model never has to learn "what if the phone is sideways." **You must train and evaluate exclusively on vehicle-frame-rotated inputs** — training on raw IO-VNBD phone-frame channels without applying the equivalent rotation would silently misrepresent what the deployed model sees.

**Window:** 2.0 s = 20 samples at 10 Hz. Stride 0.5 s at training time (75% overlap — legitimate *within* a split, catastrophic *across* splits per §6.2). Stride one window at inference.

**Channels (12 per timestep, used to derive the model's *training-time* feature engineering — note the exported model's contract input is the fixed 6-channel `accel_x/y/z, gyro_x/y/z` per §2.2; any richer feature engineering happens inside the model architecture, not as extra input channels, unless you deliberately bump the contract):**

| Group | Fields | Why |
|---|---|---|
| Linear acceleration, vehicle frame | `a_fwd`, `a_lat`, `a_up` | Primary speed-correlated signal |
| Angular rate, vehicle frame | `ω_roll`, `ω_pitch`, `ω_yaw` | Turning dynamics |
| Accel magnitude + short-window std | `‖a‖`, `σ(‖a‖)` | Vibration energy — the texture that carries speed information |
| Angular-rate magnitude + std | `‖ω‖`, `σ(‖ω‖)` | Road-roughness excitation |
| Gravity-direction stability | `Δθ_gravity` | Detects mount disturbance within the window |
| Vertical band energy | `E_up` | Suspension response |

**Target: scalar forward displacement over the window, in metres — not instantaneous speed** (see the §2.2 open-decision flag: reconcile this against the deployed contract's `velocity_mean_mps` naming before training). Displacement is the quantity the filter needs directly, and predicting-then-integrating speed re-introduces the accumulation problem the whole thesis exists to avoid.

**Label construction (synchronised subset only):** integrate the four CAN wheel-speed channels (rad/s) over the window, convert to distance via wheel radius, cross-check against CAN vehicle-speed and reject windows disagreeing beyond tolerance. **The disagreement itself becomes `label_sigma_m`** — you train with per-sample label uncertainty, not a pretence of perfect labels. Wheel radius per vehicle: `[VERIFY: derive by regressing wheel angular rate against VBOX GPS speed on straight, GNSS-clean stretches — do not look it up]`.

**Normalisation:** per-channel mean/std from the training split only, serialized into `model_manifest.json`'s `normalization` block (§2.2). Tested.

**Augmentation:** random static SO(3) rotation (arbitrary mounting), mid-window rotation discontinuity (remount, labelled synthetic), additive band-limited noise + per-channel gain jitter (different phone models), simulated bias walk (thermal drift), time-warp ±5%. **Not used: mirroring or time reversal** — a reversed drive isn't physically valid and would teach wrong dynamics.

### 6.5 Architecture (v3 PRD §14.2)

**ANCHOR-Net — one dilated temporal convolutional trunk (TCN), four heads.** Dilations 1/2/4, kernel 3, receptive field 15 timesteps (1.5 s, inside the 2 s window). Parameter target: under 50,000, int8-quantized. `[VERIFY: exact params and on-device latency after export]`.

- **Head A (mean displacement):** softplus output, ≥ 0.
- **Head B (log-variance) — the one that matters.** Trained with Gaussian NLL, explicitly optimized to know when it's uncertain. This is your primary novelty claim, measured via FR-08's calibration test, not asserted.
- **Head C (motion context, Should — S-15):** 5-way softmax (`idle`/`normal`/`rough`/`impulse`/`handling`), consumed by Kamal's noise scheduler. **Only 3 of 5 classes have real CAN-derived labels** (`idle`, `normal`, `rough`); `impulse` and `handling` have no CAN correlate. **Default: train Head C with a masked loss on the three real classes only**, and let Kamal's deterministic detectors (Hampel rejection count, remount detector) cover the other two — because this means no reported number depends on a synthetic label. A synthetic-label variant is a separate, clearly-labelled ablation, never the reported default.
- **Head D (yaw increment, Should — S-16, FR-32):** per-window yaw increment + variance, gated on the Week-5 ablation showing it beats Head-D-disabled at 5 seeds.

**Rejected alternatives, and why (carry these — a judge will ask):** an LSTM/GRU (rejected: quantizes poorly, and — critically — a recurrent hidden state carried across an outage is a hidden integrator, exactly the accumulating-error structure the thesis removes; **you still train a GRU variant as ablation row 12 and report it — if it wins, you say so and switch**); a transformer (rejected on latency/parameter budget for a 20-timestep sequence, not fashion).

### 6.6 Training procedure (v3 PRD §14.3)

| Item | Value |
|---|---|
| Framework | PyTorch → ONNX (int8 post-training quantization, calibration set from training split only). **One artifact, all heads in one graph** — matches §2.2's single exported `.onnx`. |
| Loss | `L = L_NLL + λ_c·L_context + λ_d·L_yaw`. Gaussian NLL on displacement + small L2 on log-variance (prevents variance collapse); masked cross-entropy for context; Gaussian NLL for yaw. **Run `λ_c=0` and `λ_d=0` as ablations** to confirm the auxiliary heads help rather than compete. |
| Optimiser | AdamW, lr 3e-4, cosine decay, weight decay 1e-4 |
| Batch | 256 windows, sampled so no batch is dominated by one sequence |
| Epochs | Early stopping on validation NLL, patience 15 |
| Class balance | Windows re-weighted by speed decile — otherwise the model is worst exactly where tunnels and car parks live (low-speed, high-manoeuvre) |
| Seeds | **Every reported number is mean ± std over 5 seeds. A single-seed result is not reported, ever.** |
| Determinism | Seeds, split manifests, dataset SHA-256s committed; `ml/train/run.py --config` fully reproducible from the repo |

**Fallback if the synchronised subset is too small** (the largest data risk, R-01): two-stage training — pre-train on the *unsynchronised* smartphone sequences (~58 h) using **GNSS-derived displacement as a weak 1 Hz label**, then fine-tune on whatever synchronised data exists with clean wheel-speed labels. This is exactly the mechanism §5's flywheel applies structurally, applied here to the dataset itself, and it's the same supervision arXiv:2505.18490 uses.

### 6.7 Evaluation metrics (v3 PRD §14.6, full)

| Metric | Definition | Why |
|---|---|---|
| Drift as % of distance travelled | `‖p̂(T)−p(T)‖ / ∫‖ṗ‖dt` over the outage | **The PS's own benchmark. Under 10%. Reported first, always.** |
| CTE (cross-track error) | Perpendicular distance to ground-truth path | Comparable to B4/WhONet |
| CRSE (cumulative root squared error) | Per WhONet protocol | Comparability |
| ATE (absolute trajectory error) | RMSE after rigid alignment at outage start | Standard odometry metric |
| RTE (relative trajectory error) | Error over 10/100/500 m sub-intervals | Separates "drifts slowly" from "one bad jump" |
| Error growth curve | Median/p95 horizontal error vs outage duration at 30/60/120/180 s | **The single most informative plot — shows whether error is linear or quadratic, i.e. whether the thesis is actually true** |
| Expected calibration error of σ² | Binned empirical vs predicted error, reliability diagram | FR-08. Nobody else will show this. |
| Heading error at outage end | Absolute yaw error, degrees | Also the metric Head D must move to survive |
| Map-match precision / refusal rate | Kamal's metric — you may need it for the dashboard | FR-16 |
| Integrity ROC | Detection vs false-rejection per attack family | FR-31, yours |
| Time-to-recover after reacquisition | Seconds until error falls below pre-outage level | FR-14/FR-34, Kamal's, dashboard-displayed |

**Outage protocol:** synthetic, held-out sequences, continuous ground truth, 30/60/120/180 s matching WhONet's published protocol. Start points cover the scenario mix (motorway cruise, roundabout, hard braking, sharp cornering, stop-start). **Report the per-scenario breakdown, not just the average — averages hide the roundabout.**

**The 13-row ablation table — every row is one line on the results slide (v3 PRD §14.6):**

1. Strapdown INS only (B2) · 2. +NHC · 3. +NHC+ZUPT (B3) · 4. +velocity head, fixed R · **5. +velocity head, predicted σ²→R (your primary novelty claim)** · 6. +context head → adaptive noise · 7. λ_c=0 vs >0 · 8. +Head D · 9. map matching forward-only vs fixed-lag Viterbi · 10. +road-manifold constraint · 11. +magnetic route memory · 12. GRU variant · 13. Out-of-distribution (France, Nigeria).

Rows 1-8, 12-13 are yours to run and score (`ml/bench/run_baselines.py`, `ml/eval/`); rows 9-11 need Kamal's filter — coordinate on when his components are ready enough to plug into your ablation runner.

### 6.8 The golden test set (v3 PRD §14.7)

**Almost no hackathon team does this. It is the strongest available signal that the team is serious.** 40 outage segments, frozen at end of Week 3, from test splits only, covering the scenario mix and all four durations. Each carries source sequence ID, start/end sample indices, scenario label, distance, SHA-256. Manifest at `ml/golden/manifest.json`, append-only, any change needs a PR that says why.

**Rules you bind yourself to:**
1. Never used for training, hyperparameter selection, early stopping, or architecture choice.
2. Evaluated **at most twice before the internal round** — Week-6 gate, final freeze. Every additional evaluation is overfitting-by-human.
3. **CI runs a regression gate on a 10-segment public subset on every push**: median drift regressing >5% relative fails the build. The other 30 segments are held for the two full evaluations.
4. Every reported result carries model version hash, split manifest hash, dataset SHA-256, seed list, commit hash. **A number without this provenance does not go on a slide.**
5. **If the golden set says you don't meet the PS benchmark, the slide says you don't meet it, and by how much.**

### 6.9 Failure modes and fallbacks you're responsible for (v3 PRD §14.8, filtered)

| Failure mode | Detection | Fallback |
|---|---|---|
| Model returns a physically impossible value | Kamal's bounds validator, against bounds you supply | Reject the measurement, treat as a gap, inflate covariance |
| Domain shift (Indian road, unseen phone) | Online residual monitor comparing your head's output against GNSS-derived displacement when GNSS *is* available (Kamal implements the monitor; you're the reason it exists) | Per-device scale correction learned online; if residual stays large, fall back to B3, flag device for flywheel |
| Model missing, corrupt, or unsigned | Manifest hash check at load (FR-25) | Degrade to NHC-only (FR-24) |

---

## 7. Integration & Failure-Mode Playbook

This is the comprehensive enumeration the task calls for. Every row names a concrete, standard-practice fix — not an exotic invention.

### 7.1 Dependency / environment drift

| Failure mode | Why it happens | Fix |
|---|---|---|
| `pip install torch` resolves a different wheel on your machine vs. a CI runner (CPU vs CUDA build) | PyTorch's default index serves different wheels by platform/CUDA availability, silently | **Record the exact `pip install` command used, including `--index-url` if CUDA, in `TOOLCHAIN.md` and your `README.md`.** Pin the exact version string in `requirements.txt` with `==`, generated by `pip freeze > requirements.txt` inside a fresh virtualenv (per `TOOLCHAIN.md`'s existing rule). CI installs from that file — reproducible regardless of what your local machine happened to resolve. |
| Python 3.13 vs 3.11 — some ML packages lag on new-Python support | You upgrade your local interpreter without checking | `TOOLCHAIN.md` pins **3.11.x** explicitly, with the stated reason. Don't deviate; if you must, it's a `TOOLCHAIN.md`-governed PR both people review. |
| `onnx`/`onnxruntime` minor version bump silently changes `onnx.checker` behavior or op numerics | Floating-point rounding in some op changes between minor releases | Pin exact patch versions (`onnx==1.22.0`, `onnxruntime==1.25.0` per `TOOLCHAIN.md`). **`test_contract.py`'s golden-vector test is the safety net that catches this even if a pin slips** — run it after any dependency bump, before pushing. |
| No lockfile — "works on my machine" | `requirements.txt` uses `>=` ranges, or doesn't exist | `requirements.txt` with `==` pins for everything, committed, generated fresh in a clean virtualenv. `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt` must produce byte-identical package versions for both of you. |
| Confusing training-side `onnxruntime` (Python) with Android's ONNX Runtime Mobile | They're different builds of the same project with potentially different opset support | `TOOLCHAIN.md` calls this out explicitly as "the row most likely to cause a runs-in-Python-crashes-on-Android bug." **Before any model change that touches ops beyond `Reshape`/`Gemm`, confirm the pinned `onnxruntime-android` AAR version (Kamal's side, recorded in `TOOLCHAIN.md`) supports the opset you're targeting** — message him, don't assume. |

### 7.2 Model I/O drift

| Failure mode | Why it happens | Fix |
|---|---|---|
| Window size or feature order silently changes between a training run and what's exported | Someone edits `anchornet.py`'s input handling without touching the exported contract | `generate_stub_model.py`'s constants (`WINDOW_SIZE_SAMPLES`, `FEATURE_ORDER`) are the single source of truth — your training pipeline's windowing code must import from the same constants, not redefine them locally. Any change is caught by `test_contract.py` failing to reproduce golden vectors, and by `contracts-ci.yml`'s stale-artifact check. |
| Normalization stats hardcoded separately in Python and Kotlin, and drift apart | Someone pastes the mean/std into a Kotlin file "just to get it working" instead of reading the manifest | **There is exactly one legitimate source for these numbers: `model_manifest.json`'s `normalization` block, generated by `generate_stub_model.py`.** Kamal's Kotlin code reads the manifest (or the embedded ONNX metadata) at load time — never a hardcoded constant. If you see a normalization number typed literally into a Kotlin file, that's the bug from `contracts/units.md`'s own "the bug that will happen if this file is ignored" pattern, applied to normalization instead of units — flag it immediately. |
| Mean-vs-log-variance-vs-std parametrization confusion | `velocity_log_variance` gets treated as variance, or as std-dev, somewhere downstream | The manifest states explicitly: `"consumer computes variance = exp(this) and MUST NOT treat this value as variance or std directly."` **This is exactly the kind of bug a golden vector catches numerically but a code review might miss** — if you ever change how a head's output is parametrized (e.g. switch to predicting std directly), that is a MAJOR contract bump with a new output name, never a silent reinterpretation of the same field name. |
| Batch-dimension mismatches | Python trains/evals with batch size 256; the exported graph and golden vectors use batch=1; someone forgets to drop/add the batch dim when calling from Kotlin | The contract's `input_shape` is explicitly `[1, 20, 6]` — batch fixed at 1, one inference call per window. `generate_stub_model.py`'s `make_deterministic_inputs()` and `test_contract.py`'s `_run_case` both handle the batch-dim add/drop explicitly and are the reference for how Kotlin's ONNX Runtime Mobile call must shape its input tensor. |
| Quantization changes numeric behavior after export (int8 PTQ) | Post-training quantization is not exact — a fp32 golden vector might not survive tolerance after quantizing | **Regenerate golden vectors from the quantized model, not the fp32 one** — `tolerance_abs` may need loosening for a quantized model (document why, in the same PR that changes it) but never silently; run `test_contract.py` against the actual artifact you're shipping, not an intermediate one. |

### 7.3 Units and data mismatches

| Failure mode | Why it happens | Fix |
|---|---|---|
| Gyro deg/s vs rad/s (the ~57× bug named explicitly in `contracts/units.md`) | IO-VNBD's own smartphone gyroscope columns are documented in the paper's units, which may not be rad/s — `units.md` says explicitly: "verify per-column before use; do not assume rad/s from the column name alone" | Convert at the data-loading boundary (`ml/data/`), immediately, and assert `abs(value) < 10 rad/s` on every gyro column you load — the same sanity bound `validate_replay_csv.py` enforces on the CSV side. Do this in your IO-VNBD loader **before** any downstream code touches the values. |
| km/h vs m/s | A dataset column or a partner-fleet export uses km/h | `units.md`: convert at the boundary (÷3.6), never carry km/h downstream. |
| Degrees vs radians for heading/yaw | Mixed convention between your training labels (possibly derived from a degrees-based CAN/GPS course field) and the model's internal radian convention | `units.md`: radians in all math, degrees only at UI/OSM boundaries, through one named conversion function. When you construct Head D's yaw-increment training labels from GPS course-over-ground (degrees) or CAN yaw rate, convert once, at load, and document the conversion function you used. |
| **Coordinate frame / axis convention — phone frame vs vehicle frame vs NED/ENU** | `contracts/units.md` fixes unit *magnitudes* (m/s², rad/s, etc.) but does **not** fix the axis convention — which axis is "forward," which is "up," right-handed vs left-handed, NED vs ENU for any north-referenced quantity. This is a real gap in the scaffold. | **This is an open decision — see §9's list. You must not train the real model until this is locked down with Kamal, because your training labels' sign conventions and Kamal's `AlignmentService` output must agree, or the model will silently learn the wrong sign for lateral acceleration or yaw rate.** Recommended default (state this explicitly in a new `contracts/frame_convention.md` you write together, and reference it from `model_manifest.json`): vehicle frame is right-handed, x=forward, y=left, z=up (matching common robotics/vehicle-dynamics convention, e.g. ISO 8855), NOT phone-native or NED. Heading zero = vehicle forward = local road bearing at rest. Confirm this matches what IO-VNBD's own labelled channels (`a_fwd`/wheel-speed sign convention) imply before training — don't just pick a convention in the abstract, verify it against the actual dataset columns. |
| Timestamp epoch/unit confusion | A float timestamp in seconds gets treated as milliseconds, or local time leaks in | `units.md` / `schema.json`: milliseconds since Unix epoch, UTC, always `int64`. When converting IO-VNBD's own timestamp format for any replay-CSV export tooling you write, verify explicitly against the paper's stated format — don't assume. |
| Missing-value sentinel bugs (0.0 vs empty) | A GNSS-invalid row gets `0.0, 0.0` for lat/lon instead of empty fields — `0,0` is a real coordinate off the coast of West Africa | If you ever write IO-VNBD→replay-CSV conversion tooling, this is `validate_replay_csv.py`'s explicit cross-field check (`gnss_valid=0` ⇒ all four `gnss_*` fields empty) — run that validator on anything you produce before handing it to Kamal. |
| Locale decimal-separator corruption | A CSV gets opened and re-saved in Excel on a non-US-locale Windows machine, silently swapping `.` for `,` | `schema.json`'s own documented failure mode. Never open/re-save any contract CSV in Excel — script or plain-text-edit only. If you ever touch `sample_replay.csv` or write your own CSV export tooling, run it through `validate_replay_csv.py` before committing. |
| CRLF/BOM encoding issues | Windows text editors default to CRLF; some editors add a BOM | `.gitattributes` forces LF on checkout (already in the scaffold — don't fight it). `validate_replay_csv.py` explicitly checks for BOM bytes and stray `\r`. If you write CSV-producing scripts, write with explicit `newline='\n'` / UTF-8-no-BOM in Python (`open(path, 'w', newline='\n', encoding='utf-8')`). |

### 7.4 API/contract drift

| Failure mode | Why it happens | Fix |
|---|---|---|
| snake_case (Python) vs camelCase (Kotlin) field name mismatch | The classic cross-language API bug — a Kotlin data class expecting `downloadUrl` gets a null field because the Python API sent `download_url` | **Solved structurally by `CamelModel`** (§2.4) — every Pydantic model uses `alias_generator=to_camel`, so JSON on the wire is camelCase natively. **Enforce this as a review rule: no new Pydantic model in `stub_api.py` or `backend/app/` may subclass plain `BaseModel`.** |
| Enum value mismatches | Kotlin's enum and Python's `MapRegion` diverge — a region gets added on one side and not the other | `MapRegion` is a closed enum in `stub_api.py`; adding a value is an `openapi.json`-visible, CI-diffed, `API_CONTRACT_VERSION`-bumping change (§2.6) — not something that can happen silently on either side. |
| Optional/required field disagreements | Python marks a field optional (`Optional[str] = None`); Kotlin's generated/hand-written client assumes it's always present, or vice versa | The OpenAPI spec (`openapi.json`) is generated directly from the Pydantic models — Kotlin should codegen its client from the same spec, or, if hand-written, be reviewed against it every time it's regenerated. `contracts-ci.yml`'s stale-diff check means the spec can never silently drift from the implementation. |
| Breaking a response shape without a version bump | Convenience — "it's just adding a field, no big deal" | Even additive changes get a MINOR bump per `VERSIONING.md` so the compatibility matrix stays accurate; anything that changes an existing field's name/type/presence is MAJOR, always, no exceptions, and always preceded by a message to Kamal. |
| Error-response shape not agreed | The v3 PRD (§13) specifies a structured error envelope (`{error: {code, message, field, request_id, docs}}`) that the current stub doesn't yet implement (it uses default FastAPI `HTTPException` bodies) | **Before Kamal writes error-handling code against your API, agree on the actual error envelope shape and implement it consistently across every endpoint** — this is one of §9's open items. Don't let each endpoint invent its own error shape as you add them. |

### 7.5 Versioning / release management

| Failure mode | Why it happens | Fix |
|---|---|---|
| No compatibility matrix, so nobody knows which app version needs which model contract version | Versions tracked "in someone's head" | `VERSIONING.md`'s compatibility matrix table — keep it updated every time you bump either `model_io` or `backend_api`'s contract version. It currently has one row (`0.1.0` → `1.0.0`/`1.0.0`); add a row every real bump. |
| App silently tries to load an incompatible model instead of refusing | The refusal logic is Kamal's, but it depends on you setting `minSupportedContractVersion` correctly on every `/v1/model/version` response | §2.6 — when you bump the model contract's MAJOR version, update the corresponding field on the API response in the same PR. This is the single point of failure for the whole refuse-don't-try-anyway rule (FR-24). |
| No changelog discipline | Contract changes happen in code without a paper trail explaining why | Every contract-touching PR description states which version number moved and why (semver rule from `VERSIONING.md`), and the pre-PR message to Kamal (§2.6) is the informal changelog for a two-person team — but still put it in the PR description too, so it's searchable later. |

### 7.6 Git / process risks

| Failure mode | Why it happens | Fix |
|---|---|---|
| Two people editing the same file | Both of you touch `contracts/` occasionally | `contracts/` is explicitly the one shared folder (per `README.md`) — touch it only with a heads-up message first, per `VERSIONING.md`. Everything else (`ml/`, `backend/` vs `core/`, `android/`, `edge/`) is exclusively owned per track, so simultaneous edits shouldn't happen there at all. |
| Merge conflicts | Long-lived branches diverge | Branch-per-feature, short-lived, frequent merges — **recommend merging to `main` at least every other day**, more often during integration weeks (P2 onward per the v3 PRD's roadmap), so drift never accumulates past a day or two. |
| Unreviewed pushes to `main` | No branch protection configured | `README.md`'s own setup instruction: protect `main`, require PR review + the `contracts-ci` status check before merge. **Set this up in the very first session, before any real code lands** — it's step 3 of the shared setup checklist. |
| Infrequent syncing lets drift accumulate | Working in isolation for a week before merging | Same fix as merge conflicts — daily-to-every-other-day merge cadence, plus the explicit `VERSIONING.md` pre-change message for anything touching `contracts/`. |
| Secrets accidentally committed | An API key, a `.env` value, a signing key pasted into a script during a debugging session | `.env.example` committed with variable names and **no values**; real `.env` gitignored from commit one. **The model-signing private key never touches a developer machine — it lives in the CI secret store only** (v3 PRD §15.3) — you never need it locally, don't request it. Install a `gitleaks` pre-commit hook and CI job in Week 1, before there's history to clean. |
| Large binary files bloating the repo | A trained checkpoint or a `.pt` file gets `git add`ed by habit | `.gitignore` already excludes `*.ckpt`, `*.pt`, `checkpoints/`, `runs/`, `wandb/`, `data/raw/`, `*.osm.pbf` — **real trained weights ship through the model registry (`GET /v1/model/version`'s `downloadUrl`), never committed.** Only the small stub `.onnx` in `contracts/model_io/` is ever committed as a binary — check `git status` before any commit that touches training output, and if you see a `.ckpt` or `.pt` staged, that's a mistake to undo, not a file to force-add. |

### 7.7 CI / testing discipline

| What | Command | When | Owner |
|---|---|---|---|
| Model contract self-consistency + golden vectors | `pytest contracts/model_io/test_contract.py -v` | Before every push touching `generate_stub_model.py` or anything that changes model weights/architecture | You |
| Replay CSV validation | `python contracts/replay_csv/validate_replay_csv.py <file>` | Before handing any CSV you generated (e.g. IO-VNBD conversion output) to Kamal | You, when applicable |
| OpenAPI diff check | `python contracts/backend_api/stub_api.py` then `git diff --exit-code -- openapi.json` | Before every push touching `stub_api.py` or `backend/app/` request/response models | You |
| **Android instrumented golden-vector test** | *Not yet built.* Stubbed as a commented-out job (`android-contract-check`) in `.github/workflows/contracts-ci.yml` | — | **Kamal's to-build item, not yours** — but you should know it will exercise your golden vectors against ONNX Runtime Mobile, so keep `tolerance_abs` realistic for a quantized on-device runtime, and tell him if you tighten or loosen it. |
| `gitleaks` secret scan | Pre-commit hook + CI | Every commit, from Week 1 | Shared setup, either of you |
| Golden-set regression gate (10-segment public subset) | Part of `ml/eval/`'s CI integration | Every push touching the model or training pipeline | You |

---

## 8. Week-by-week checklist

*Consistent with v3 PRD §20's phasing (P0–P4), scoped to your deliverables. Explicit sync points with Kamal are marked **SYNC**.*

### Week 1 (P0 — Foundations)

- [ ] **Day 1, hour 1:** Measure the size of the synchronised V+S subset (v3 PRD §I.2, R-01). This gates the entire supervised plan — do this before writing any model code.
- [ ] **Day 1:** Resolve whether IO-VNBD has effectively one vehicle or three (v3 PRD §I.1, R-02) — read Tables 3–4 directly. Whatever the answer, write down exactly what the split protocol can honestly claim.
- [ ] **Day 1:** Retrieve the official PS text directly from sih.gov.in and reconcile (R-18) — shared with the generalist/whole team, but flag anything that changes your scope.
- [ ] Set up `requirements.txt` with exact pins per `TOOLCHAIN.md`; record your exact `pip install torch ...` command including index-url.
- [ ] Run `python contracts/model_io/generate_stub_model.py` and `pytest contracts/model_io/test_contract.py` locally — confirm they pass on your machine before you touch anything.
- [ ] Run `uvicorn contracts.backend_api.stub_api:app --reload` locally, hit `/docs`, confirm the four endpoints respond.
- [ ] Install `gitleaks` pre-commit hook.
- [ ] **SYNC:** with Kamal, fill in `TOOLCHAIN.md`'s Kotlin/AGP/Gradle/onnxruntime-android rows together.
- [ ] **SYNC:** lock down the phone-frame/vehicle-frame/NED-ENU axis convention (§7.3, §9) — write it into a new `contracts/frame_convention.md`.
- [ ] **SYNC:** resolve the `velocity_mean_mps`-vs-displacement naming tension (§2.2, §9) before any real training starts.
- [ ] Set branch protection on `main`: require PR review + `contracts-ci` status checks.
- [ ] Build IO-VNBD data loaders (`ml/data/`) with the gyro-unit sanity assertion applied at load.

### Week 2 (P0 continued)

- [ ] Split protocol implemented as code, split manifests committed (`ml/splits/*.json`).
- [ ] `SequenceWindower` + `test_no_cross_sequence_windows` passing.
- [ ] B1 (constant-velocity extrapolation) implemented and running against held-out data.
- [ ] Coordinate with Kamal on B2/B3 (his `reference/anchor_ref/` Python oracle) — you need these for the ablation table; confirm interface early.
- [ ] `ml/bench/run_baselines.py` skeleton exists, even if only B1 runs.

### Weeks 3–5 (P1 — The thesis)

- [ ] ANCHOR-Net trunk + Head A + Head B implemented (`ml/models/anchornet.py`), matching the exact input/output contract in `generate_stub_model.py`.
- [ ] Normalisation fitted on train split only, tested (`test_normaliser_fitted_on_train_only`).
- [ ] Head C (context) with masked loss on the 3 CAN-derived classes.
- [ ] Head D (yaw increment) trained, ablation harness ready.
- [ ] **End of Week 3: golden set frozen** (v3 PRD §14.7) — `ml/golden/manifest.json` committed, append-only from here on.
- [ ] Full ablation runner covering rows 1-8, 12-13 of the 13-row table.
- [ ] GRU variant trained and measured (rejected-alternative row).
- [ ] **Week-5 gate (hard checkpoint, v3 PRD §20.1): present the velocity-head-vs-B3 result to the full team.** If ≥20% relative beat with acceptable calibration → proceed as planned. If beats B3 but <20% or poor calibration → cut Head C/D that day, demote confidence language, tell Kamal his `FusionService` falls back to fixed `R`. If it doesn't beat B3 at all → the pivot in v3 PRD §20.1 applies: the demo pivots to B3 as the shipped system, and FR-29/FR-31 (both independent of ML) become the headline. **This decision is not yours alone to sit on — report it to the team the same day the numbers come in.**
- [ ] **SYNC (whenever real normalization stats are computed, likely end of Week 3–4): regenerate `model_manifest.json` with real `NORM_MEAN`/`NORM_STD` via `generate_stub_model.py`. This is a contract-relevant change even if it's not a MAJOR version bump (values changed, not shape) — message Kamal, since his app needs the new manifest before the model is useful, and re-run the golden vectors together with him watching if possible.**
- [ ] **SYNC:** when the real-weights model is first exported (replacing the stub), message Kamal immediately — this unblocks his `ModelRunner.kt` integration testing against something real instead of the linear stub.

### Weeks 6–7 (P2 — On-device, your parts)

- [ ] ONNX export + int8 quantization pipeline (`ml/export/`).
- [ ] Regenerate golden vectors from the quantized model; adjust `tolerance_abs` if needed, document why.
- [ ] `ml/export/sign_manifest.py` — SHA-256 hash + sign every exported artifact; confirm the public key Kamal pins matches what you sign with.
- [ ] **Week-6 gate: on-device latency benchmark** (shared with Kamal — v3 PRD §14.5) — coordinate on getting your exported model onto his test device.
- [ ] FR-31 integrity bench: attack injector + ROC scoring (`ml/integrity/`), tested against Kamal's `ChiSquareGate.kt` once it exists.
- [ ] Backend `POST /v1/telemetry/labels` implemented for real (currently a stub echoing counts) — bounds validation, consent check (server-side 403 re-check per FR-23).
- [ ] Backend model registry (`GET /v1/model/version`) serving real signed artifacts, not stub data.

### Week 8 (P3 — Binding and polish, your parts)

- [ ] Web evaluation dashboard API (`backend/app/dashboard_api/`) serving the ablation table, calibration reliability diagram, integrity ROC.
- [ ] `web/` frontend rendering the above (build it yourself if no one else is available).
- [ ] Backend map-extract hosting (`GET /v1/map/extract`) serving real `.osm.pbf` extracts for the demo corridors, coordinated with Kamal's Week-4 corridor geometry audit.

### Final week (P4 — Freeze and evidence)

- [ ] **Second and final golden-set evaluation** (the second of your two permitted full evaluations, per v3 PRD §14.7 rule 2).
- [ ] Results sheet with full provenance: model version hash, split manifest hash, dataset SHA-256, seed list, commit hash — for every reported number.
- [ ] Confirm `VERSIONING.md`'s compatibility matrix table is accurate and complete for whatever versions are actually shipping.

---

## 9. Open decisions — flagged, not silently resolved, must be settled with Kamal in Week 1

1. **Axis / frame convention.** `contracts/units.md` fixes unit magnitudes but not the coordinate frame — which axis is forward, up-vs-down sign, NED vs ENU for any north-referenced quantity, and the exact rotation `AlignmentService` applies to go from phone frame to vehicle frame. **Recommended default: right-handed vehicle frame, x=forward, y=left, z=up (ISO 8855-style), confirmed against IO-VNBD's own labelled sign conventions before training starts.** Write the agreed convention into a new `contracts/frame_convention.md` and reference it from `model_manifest.json`.
2. **`velocity_mean_mps` vs. displacement semantics.** The current scaffold's model output is named and described as mean *speed* (m/s); v3 PRD §14.1 specifies the training target as scalar *displacement* over the window (metres). Decide which the real model will output, rename the field if needed, and bump `contract_version` accordingly — before real training, not after.
3. **The error-response envelope shape.** v3 PRD §13 specifies a structured error format (`{error: {code, message, field, request_id, docs}}`) not yet implemented in the current `stub_api.py` (which uses default FastAPI `HTTPException` bodies). Agree on and implement the real shape before Kamal writes error-handling logic against it.
4. **The fuller §13 API surface vs. the current 4-endpoint stub.** The v3 PRD's §13 describes additional endpoints (`/maps/regions`, `/devices/enrol`, `/labels/batch`, `/magsig/batch`, `/fleet/trips`) that don't exist in the current scaffold. Decide together which of these are actually needed for the hackathon demo vs. deferred, so you're not building unused surface area or, worse, so Kamal isn't assuming an endpoint exists that you haven't built.
5. **Model bounds for FR-24's implausibility check.** You own the numbers (e.g. "displacement > X m in a 2 s window is impossible"); document them explicitly in a place Kamal can read (extend `model_manifest.json` or a sibling contract file) rather than communicating them informally.

---

## 10. What "done" looks like before you push, every time

Before any push touching `contracts/model_io/` or `contracts/backend_api/`:

1. Did you change the generator script's source of truth, not the generated output directly?
2. Did you re-run the generator and commit its output?
3. Does `pytest contracts/model_io/test_contract.py` (or the equivalent openapi diff check) pass locally?
4. Is this a MAJOR/MINOR/PATCH change per `VERSIONING.md`, and did you bump the right version number?
5. If MAJOR: did you message Kamal *before* writing the code, not after?
6. Would `contracts-ci.yml` pass on this push? (Run the regeneration + diff check locally before pushing if you're not sure.)

If the answer to any of these is "no" or "not sure," fix it before the push, not after.
