# PRD — ANCHOR: On-Device Engine + Android App Track (Kamal)

**Parent document:** `/home/claude/ANCHOR-PRD-v3.0-FINAL.md` ("the v3 PRD"), Project ANCHOR, SIH 2026 PS 26168. This document is a track-scoped extraction of the v3 PRD plus a full integration/failure-mode playbook and a week-by-week plan. It is written so a fresh Claude Code session, given only this file, can start building the Kotlin engine and Android app without needing the v3 PRD or the other track's document open.

**Owner:** Kamal. **Scope:** the error-state EKF (NHC, ZUPT, bias states), phone/vehicle alignment, the GNSS quality monitor, OSM map matching + the road-manifold constraint, magnetic route memory, RTS smoothing, the replay harness (Kotlin side), the Android app (map/marker/confidence ring/UI), and the edge/CLI engine variant.

**Companion document:** `/home/claude/PRD-ML-BACKEND.md`, owned by Harshit — covers ANCHOR-Net (the velocity + variance model), the leakage-safe training pipeline, the evaluation harness + golden set + baselines, the GNSS integrity bench's attack injector, and the Python/FastAPI backend. **Everything in that list lives in Harshit's document, not here** — if you find yourself training a model or writing FastAPI routes, stop; it's out of scope for this track.

**The seam between the two tracks is `/home/claude/scaffold/`** — a tested, running set of contract files that fix the exact I/O between what you build and what Harshit builds. §2 of this document restates that contract precisely for your side. Read `/home/claude/scaffold/README.md` once, at the start, regardless.

---

## 0. How to read this document

This is carved from the v3 PRD (version 3.0, FINAL). Every FR, acceptance criterion, and risk quoted below is reproduced from that document, not paraphrased, except where marked `[adapted]`. Numbers marked `[VERIFY]` are unmeasured — do not treat them as fact until measured on your target device. §§ references (e.g. "§10.2") point at the v3 PRD's own section numbers.

**What is deliberately NOT in this document** (it's all in Harshit's): the ANCHOR-Net architecture and training procedure, the IO-VNBD leakage-safe split protocol, the evaluation metrics and ablation methodology, the golden test set's construction and evaluation-frequency rules, the baseline implementations (B1-B5) beyond what you need for your own filter's B2/B3 reference, the FastAPI backend's routes and business logic, and the flywheel retraining pipeline. You own the *filter* that consumes the model's output and the *app* that renders it; you do not own the model or the server.

---

## 1. Shared context (condensed from v3 PRD Part II §§1-7)

### 1.1 The problem, in one paragraph

A phone loses GNSS in tunnels, multi-level parking, and urban canyons. Existing nav apps freeze the marker, then teleport it on reacquisition. Naive IMU dead-reckoning fails because double-integrating accelerometer noise for position produces error that grows with the *square* of time (v3 PRD §1.2). The arithmetic, both halves, matters because it drives your filter's design directly:

**Along-track, from accelerometer bias:** at a deliberately optimistic 1 mg (0.0098 m/s²) bias, error is ≈0.49 m at 10 s, ≈4.4 m at 30 s, ≈17.6 m at 60 s, ≈159 m at 180 s.

**Cross-track, from gyro yaw bias — the half teams forget:** at 60 km/h with a residual yaw bias of 0.1°/s, cross-track error over 60 s is `16.7 × 0.001745 × 60² / 2 ≈ 52 m` — **half the entire 100 m budget, from heading alone.** Three consequences that shape your filter directly: (1) gyro bias must be an estimated state, re-observed at every opportunity — this is what makes ZUPT (FR-26) load-bearing, not a nicety; (2) the magnetometer is not a heading rescue inside a steel vehicle body — the road bearing from the offline map is a far better heading constraint; (3) the road-manifold constraint (FR-29) deletes the cross-track term entirely inside a corridor — this is why it's a headline feature, not a refinement.

ANCHOR's bet: instead of integrating acceleration, learn vehicle speed from IMU vibration texture (Harshit's model — a *perception* problem, non-accumulating), then fuse that speed estimate — with a **calibrated per-window uncertainty** — into your filter, which also understands road geometry, so a tunnel becomes a line rather than a plane.

### 1.2 The one-sentence thesis (v3 PRD §3.1)

> "ANCHOR teaches a phone to feel how fast a vehicle is moving, and to know how sure it is — so that when the satellites disappear in a tunnel, the map keeps moving correctly instead of freezing, and says so when it cannot."

### 1.3 Personas (v3 PRD §7.1, verbatim, condensed) — these drive your UI and filter tuning directly

- **P1 — Ravi**, 24, delivery rider, mid-range 4 GB Android, 3-year-old chipset, degraded battery, prepaid/throttled data, limited English (**reads icons faster than text**), phone in a handlebar mount **that gets knocked out of alignment several times a shift.** → alignment must run continuously (FR-06), not once; no connectivity assumption; battery-conscious; icon-first UI.
- **P2 — Sunita**, 41, ambulance driver, hill roads, long dead stretches, extreme time pressure. → **needs the confidence ring more than anyone** — a known-bad position lets her fall back on judgement; an unknown-bad one is dangerous. Must work with zero data connection for an entire journey.
- **P3 — Arun**, 35, fleet ops manager, buys on evidence — trip-level drift statistics, disputed-delivery audit trail (his tooling is mostly Harshit's dashboard/fleet API, but your `TripExporter` and `pose`/`mode_event` data model feed it).

### 1.4 Why your filter needs a calibrated variance, not just a point estimate (v3 PRD §3.2)

A Kalman filter combines information optimally but cannot manufacture information that isn't there. The reason your `VelocityUpdate` consumes `R = σ² × trust_factor` rather than a fixed noise value is that Harshit's model is trained to know when it's uncertain (Gaussian NLL loss on log-variance) — treat this as load-bearing, not decorative: **doubling the model's predicted `σ²` must halve the state correction magnitude** (this is literally FR-11's acceptance test). If you ever find yourself hand-tuning a fixed `R` for the velocity update "because it's easier," you've defeated the single biggest differentiator the project has over a point-estimate baseline.

### 1.5 The moat, and why your filter has none — deliberately (v3 PRD §4.4)

| Component | Clonable in a week? | Verdict |
|---|---|---|
| **ESKF + NHC + ZUPT pseudo-measurements** | **Yes. Textbook. Three days for a good team.** | **No moat. Yours, and we claim none.** |
| **HMM map matching on OSM** | **Yes. Newson–Krumm is published with open implementations.** | **No moat. Yours.** |
| The velocity head itself | Yes, now — published by AVNet (2025) | No moat (Harshit's) |
| The leakage-safe training protocol | No — 4-6 weeks | Moat (Harshit's) |
| Calibrated uncertainty | No — everyone skips this | Moat (Harshit's) |
| Data + magnetic flywheel | No — needs users | The only growing moat |

**Design consequence for you, stated explicitly in the v3 PRD:** "Since half the components have no moat, we deliberately do not spend scarce weeks re-deriving a Kalman filter. We implement the standard one carefully, test it against synthetic ground truth, and spend the saved time on the split protocol, the calibration, the road-manifold constraint and the flywheel." **Your job is not to invent filter novelty — it's to implement the standard ESKF correctly, rigorously tested, and put your genuine engineering effort into the three things that *are* differentiated on your side: the road-manifold constraint (FR-29), magnetic route memory (FR-30), and the fact that one compiled artifact runs on a phone, in replay, and on a 200 Hz edge IMU (§5 below).**

---

## 2. The integration contract, restated for your side

Everything in this section is sourced from `/home/claude/scaffold/`, read directly from the files, not summarized from memory. **Treat every path, field name and value below as exact** — if you ever see a discrepancy between this section and the actual scaffold file, the file wins; re-read it.

### 2.1 What you own in `contracts/`

You own `contracts/replay_csv/` — you write the schema and the generator that produces `sample_replay.csv`; Harshit's side reads it (mainly to validate any IO-VNBD conversion tooling he writes against your schema). You do **not** own `contracts/model_io/` or `contracts/backend_api/` — you consume both, read-only, and never hand-edit their generated output.

### 2.2 `contracts/model_io/` — what you consume, and exactly how

**Never hand-edit anything under `contracts/model_io/`.** If the contract needs to change, that's a message to Harshit and a change to *his* generator, not a local Kotlin-side workaround.

The contract, exactly as it exists today (`contract_version` **1.0.0**):

- **Model file:** `anchor_net_stub.onnx`, currently a trivial two-Gemm linear stub (real weights replace this later, same file name, same interface).
- **Input tensor:** name `imu_window`, shape `[1, 20, 6]` (`[batch, time, features]`), dtype float32.
- **Window:** 20 samples = 2.0 s **at 10 Hz.** This is the model's native rate — **your engine's raw IMU sampling runs at 100-200 Hz or whatever the device natively supports (FR-01), and you must decimate down to 10 Hz with proper anti-aliasing (FR-03) before assembling the 20-sample window** the model expects. Feeding the model raw-rate data, or a window of the wrong length, is a silent contract violation the golden vectors will not catch (they test the model, not your windowing code) — you need your own unit tests for windowing correctness.
- **Feature order — fixed, positional, exactly this, nothing inferred from names:**
  ```
  accel_x, accel_y, accel_z,   # m/s², vehicle-frame-aligned
  gyro_x,  gyro_y,  gyro_z,    # rad/s, vehicle-frame-aligned
  ```
  **"Vehicle-frame-aligned" is your responsibility, not the model's.** Your `AlignmentService` (roll/pitch from gravity per FR-04, yaw from motion per FR-05) must rotate raw phone-frame samples into the vehicle frame **before** they're assembled into the window and passed to `ModelRunner`. The model has never seen phone-frame data and was never trained to handle it — passing unrotated samples will not throw an exception, it will silently produce a plausible-looking but wrong velocity estimate. This is exactly the kind of bug §7.3 below exists to catch early.
- **Outputs**, two heads, each `[1, 1]` float32:
  - `velocity_mean_mps` — m/s.
  - `velocity_log_variance` — natural-log of variance. **You must compute `variance = exp(velocity_log_variance)` before using it anywhere. Never treat the raw output as variance or as std-dev.** This feeds `FusionService.VelocityUpdate`'s `R = σ² × trust_factor` per FR-11.
- **Normalization is NOT baked into the ONNX graph** — `model_manifest.json`'s `normalization` block states `"applied_by": "caller, before inference"`. **This means `ModelRunner.kt` must read `mean`/`std` from the manifest (or the ONNX file's own embedded metadata) and apply `(raw - mean) / std` per-feature to the assembled window before calling `session.run()`.** The stub's current mean/std are `[0,0,0,0,0,0]`/`[1,1,1,1,1,1]` (a no-op), so this stage is invisible right now — **it will not stay invisible once Harshit ships real normalization stats, and if your `ModelRunner` hardcodes a no-op instead of reading the manifest, the model will silently receive un-normalized input the moment real stats ship.** Build the manifest-reading path now, even while it's a no-op, so there's nothing to retrofit later.
- **Opset:** pinned to **17**. Before you add `onnxruntime-android` to Gradle, confirm the exact AAR version you pin supports opset 17 for the ops the stub uses (`Reshape`, `Gemm` — both old and stable, low risk for the stub). **Re-check this the day Harshit's real model architecture ships**, since a TCN with GroupNorm, GELU, or int8 QDQ ops may need a newer opset than the stub does — that would be a `contracts/model_io` MAJOR bump on his side, and a corresponding AAR version bump on yours; he messages you before writing that code, per `VERSIONING.md`.
- **Embedded metadata:** the same contract fields are written into the ONNX file's own `metadata_props`. **Read this at model-load time and sanity-check it against your app's expected contract version before trusting the file** — this is literally what FR-25's "loading a model whose hash is not in the signed manifest is refused" and the `VERSIONING.md` compatibility-refusal rule require you to do, and it works without a network call since the metadata travels inside the file itself.

**⚠ Open naming/semantics tension — flag it, watch for Harshit's resolution, don't silently assume.** The current manifest names the outputs `velocity_mean_mps`/describes it as "predicted forward speed" — but v3 PRD §14.1 specifies the actual training target as scalar forward *displacement* over the window (metres), specifically because predicting speed and then integrating it re-introduces the accumulation problem the whole project exists to eliminate. **Do not assume which one you're getting until Harshit confirms it explicitly (§9's open-decisions list) — your `VelocityUpdate.kt` measurement-model math is different depending on the answer** (a displacement-per-2s-window measurement is a different Kalman update than an instantaneous-speed measurement). If this resolves as a rename, it's a MAJOR contract bump per `VERSIONING.md` and Harshit messages you before writing the training code that assumes it — but build your `VelocityUpdate` code so the distinction is a one-line change (a named constant/function for "what does this head's output mean, physically"), not something scattered through the fusion math.

### 2.3 `contracts/model_io/golden_vectors/` — your responsibility to close the loop on

Three fixed vectors: `stationary.json`, `accelerating_straight.json`, `random_vibration.json`, each with `input` (`[20][6]`), `expected_output` (both heads), `tolerance_abs` (currently `1e-4`).

**This is the anti-drift mechanism proving the exact same `.onnx` file produces the exact same numbers on ONNX Runtime Mobile as on Python `onnxruntime`.** Harshit's `test_contract.py` checks this on the Python side. **The Android-side equivalent does not exist yet — it is explicitly stubbed as a commented-out job in `.github/workflows/contracts-ci.yml`, and it is yours to build**, not his:

```
# android-contract-check:
#   runs-on: windows-latest
#   steps:
#     - uses: actions/checkout@v4
#     - name: Instrumented contract test
#       run: ./gradlew :app:connectedContractCheckDebugAndroidTest
```

**Build this as a real instrumented Android test** (§6's checklist has it as a concrete Week-2/3 task): load `anchor_net_stub.onnx` via ONNX Runtime Mobile, run each `golden_vectors/*.json` input through it, assert both outputs match `expected_output` within `tolerance_abs`. Also, per the placeholder's own comment, parse `contracts/replay_csv/sample_replay.csv` with your own CSV reader and assert row/column counts and a few spot values against the schema. **Uncomment the `android-contract-check` job in `contracts-ci.yml` once this exists**, so it actually runs in CI rather than sitting as a to-do forever.

**Run the golden-vector test before trusting any new model file — every time a new `.onnx` lands from Harshit, before you wire it into a build you'll demo from.** This is the practical form of "must run the golden-vector test before trusting a new model file" from the task brief: a model that passes golden vectors on his machine but fails on your device's actual ONNX Runtime Mobile build has found a real cross-platform numeric drift bug, and you want to find that during a routine check, not during a rehearsal.

### 2.4 `contracts/replay_csv/` — the schema you own

**Generator:** `contracts/replay_csv/make_sample_csv.py` — never hand-edit `sample_replay.csv` directly; change the generator and re-run it. **Validator:** `contracts/replay_csv/validate_replay_csv.py` — run this on any CSV before it's used for replay, and as a CI check.

**The schema, exactly (`contract_version` 1.0.0), 15 columns, this exact order:**

| # | Column | dtype | Unit | Notes |
|---|---|---|---|---|
| 1 | `timestamp_ms` | int64 | ms since Unix epoch, UTC | Strictly increasing within a sequence; no two rows share a timestamp |
| 2 | `accel_x` | float64 | m/s² | Vehicle-frame-aligned if alignment has run, else raw phone-frame |
| 3 | `accel_y` | float64 | m/s² | |
| 4 | `accel_z` | float64 | m/s² | |
| 5 | `gyro_x` | float64 | rad/s | **NOT deg/s** |
| 6 | `gyro_y` | float64 | rad/s | |
| 7 | `gyro_z` | float64 | rad/s | |
| 8 | `mag_x` | float64 | microtesla | |
| 9 | `mag_y` | float64 | microtesla | |
| 10 | `mag_z` | float64 | microtesla | |
| 11 | `gnss_valid` | int64 | 0 or 1 | 1 iff this row carries a real GNSS fix |
| 12 | `gnss_lat` | float64 or empty | decimal degrees, WGS84 | **Empty, never 0.0, when `gnss_valid=0`** |
| 13 | `gnss_lon` | float64 or empty | decimal degrees, WGS84 | Same rule |
| 14 | `gnss_speed_mps` | float64 or empty | m/s | Same rule |
| 15 | `gnss_course_deg` | float64 or empty | degrees, 0-360, clockwise from true north | Same rule |

**Encoding: UTF-8, no BOM. Line ending: LF only (`.gitattributes` enforces this on checkout). Delimiter: comma. Decimal separator: `.`. Header row present. Missing-value representation: empty string between commas (e.g. `,,`) — never `0`, never `NaN`, never the literal string `"null"` or `"NA"`.**

**The four documented failure modes this schema exists to prevent (verbatim from `schema.json`'s `known_failure_modes_this_schema_prevents`):**
1. Excel re-saving on a non-US-locale Windows machine silently swaps the decimal separator to a comma — a delimiter collision. Never open/re-save in Excel.
2. Windows editors defaulting to CRLF — some naive CSV parsers (including a naive Kotlin `BufferedReader` split) include a trailing `\r` in the last field of every row.
3. Gyro in deg/s instead of rad/s — the ~57× bug, silently diverges the filter rather than crashing.
4. Using `0.0` instead of empty for a missing GNSS fix — `0,0` is a valid coordinate (off West Africa) and silently corrupts map matching.

**Your `CsvReplaySource` (Kotlin) must parse this exact schema — this exact column order, these exact names, these exact null-handling semantics — and it must implement the equivalent of `validate_replay_csv.py`'s checks natively (or shell out to the Python validator during development/CI) so a malformed CSV fails loudly rather than silently corrupting a replay run.**

### 2.5 `contracts/backend_api/` — what you consume from the network layer

**Generator (not yours):** `contracts/backend_api/stub_api.py`. **You never hand-edit `openapi.json`** — codegen a Kotlin client from it, or hand-write one reviewed against it every time it changes.

**Every endpoint that exists today, exactly, base path `https://.../v1`:**

| Method | Path | Query/Body | Response (camelCase on the wire) |
|---|---|---|---|
| `GET` | `/v1/health` | — | `{status, apiContractVersion}` |
| `GET` | `/v1/map/extract` | `region` ∈ `{delhi_ncr, hill_corridor, uk_metrics}` | `{region, mapVersion, downloadUrl, sha256, sizeBytes, updatedAt}` |
| `GET` | `/v1/model/version` | — | `{modelVersion, contractVersion, minSupportedContractVersion, downloadUrl, sha256, sizeBytes, publishedAt}` |
| `POST` | `/v1/telemetry/labels` | `{deviceIdHash, pairs: [{imuWindow, displacementM, windowDurationS, deviceModel, appVersion, contractVersion}]}` | `{accepted, rejected, rejectionReasons, batchId}` |

**Wire format: camelCase, always** — every response model on the Python side subclasses `CamelModel` (`alias_generator=to_camel`), so your Kotlin/Retrofit/kotlinx.serialization data classes should use `downloadUrl`, `sizeBytes`, etc. directly, matching the JSON verbatim, with no manual field-name translation layer needed on either side.

**Note these four are what exists and runs today; the v3 PRD's own §13 describes a fuller intended surface** (`/maps/regions`, `/devices/enrol`, `/labels/batch`, `/magsig/batch`, `/fleet/trips`, etc.) that hasn't been built yet. Treat anything from §13 not in the table above as a roadmap item to confirm with Harshit before assuming it exists — **don't write a Kotlin client method against an endpoint that isn't in the current `openapi.json`.**

**This is your network client's entire consumption surface for the whole project.** Remember the architectural constraint this implies: **FR-22 requires zero network access for the entire trip** — none of these four endpoints may ever be called synchronously from inside the positioning loop. `GET /v1/model/version` and `GET /v1/map/extract` are pulled opportunistically when connectivity exists (app startup, background check), never awaited mid-trip; `POST /v1/telemetry/labels` is an opt-in, batched, deferred-to-Wi-Fi upload, never inline with anything real-time.

### 2.6 `contracts/VERSIONING.md` — the rule, restated for what you specifically must do

Three independent version numbers exist; you're the enforcement point for two of them:

- **`model_io` contract version** — you declare `MIN_SUPPORTED_MODEL_CONTRACT_VERSION` (or per the exact name in `VERSIONING.md`, `MIN_SUPPORTED_CONTRACT_VERSION`) as a constant in your app. **This is the single most important line of code for the whole compatibility story:**
  ```
  If model.contract_version (from the manifest, or ONNX metadata_props) is not
  semver-compatible with MIN_SUPPORTED_CONTRACT_VERSION:
      REFUSE to load. Fall back per FR-24. Log exactly why. Never "try anyway."
  ```
  Write this check once, in one place (`ModelManifest.kt` or `ModelRunner.kt`'s load path), and make sure every code path that loads a model — first launch, OTA update, replay harness model swap — goes through it. **A model that fails this check is not "probably fine" — refuse it, unconditionally, every time.**
- **`backend_api` contract version** — your network client should fail a build-time or startup check against an incompatible major version, "not a runtime crash three screens into the demo" (v3 PRD §13, quoted). At minimum, check `GET /v1/health`'s `apiContractVersion` (or the OpenAPI spec's own version) against what your client was generated/written against, on app startup, and surface a clear error state rather than letting mismatched requests fail confusingly deep in the call stack.
- **Model weights version** — every `GET /v1/model/version` response includes a `modelVersion` field that changes far more often than `contractVersion`. Log it into every exported trip trace, per FR-25 ("a running engine... carries the model version hash").

**The rule that actually prevents disasters: expect a message from Harshit *before* he writes code against a changed contract, and read it before you assume the old contract still holds.** If you ever notice `openapi.json`, `model_manifest.json`, or `schema.json`'s `contract_version` field has changed without a message reaching you first, treat that as a process failure worth raising immediately, not something to quietly work around.

### 2.7 What you own vs. what you must never touch

**You own, and are the only one who writes to:** `contracts/replay_csv/make_sample_csv.py` (and `sample_replay.csv`), `core/` (the whole Kotlin engine), `android/` (the app), `edge/` (the CLI). If Harshit needs a change to the replay CSV schema, that's a message to you and a change you make, governed by `VERSIONING.md`.

**You never touch:** `contracts/model_io/generate_stub_model.py` or anything it generates, `contracts/backend_api/stub_api.py` or `openapi.json`, `ml/`, `backend/`, `reference/` (Harshit's oracle — you consume its golden vectors for filter regression testing but don't write to it).

---

## 3. Your scope, in MoSCoW terms (v3 PRD §6.1, filtered to your items)

| ID | Item | Priority | In demo? | Note |
|---|---|---|---|---|
| S-02 | 15-state error-state EKF with NHC and ZUPT | **Must** | Yes | Visible as smooth motion and no creep at traffic lights. |
| S-03 | Phone→vehicle alignment (roll/pitch from gravity, yaw from motion) + remount detection | **Must** | Yes | Shown as a 5 s "calibrating" state. |
| S-04 | GNSS quality monitor, quality-weighted fade, hysteresis, chi-square gate | **Must** | Yes | The `GNSS / FUSED / DR` pill. |
| S-05 | Offline OSM map matching (fixed-lag HMM) with confidence-gated snapping | **Must** | Yes | The amber refusal indicator. |
| S-06 | **Replay harness** — recorded CSV through the live engine as if it were sensors | **Must** | Yes | **This *is* the demo mechanism. Build it first, in P0.** |
| S-07 | Android app: map, marker, confidence ring, mode pill, drift counter, context label | **Must** | Yes | — |
| S-09 | **Road-manifold constraint** | **Must** | Yes | Headline 1. §5.2 of the v3 PRD. Your genuine differentiation. |
| S-10 | **Edge/CLI engine** consuming CSV or serial IMU at arbitrary rate | **Must** *(was Should)* | One slide | The PS names it as a deliverable. Nearly free under the Kotlin decision — one Gradle module, two artifacts. |
| S-12 | RTS backward smoothing on reacquisition | **Must** | Yes | The trail slides into place. Textbook; not claimed as novelty. |
| S-13 | Provenance hashing on every trace, plot and slide number | **Must** | Yes (printed sheet) | Discipline, not USP. |
| S-14 | **Magnetic route memory** | **Should** | Yes if ready | Headline 2. Cut before S-01/S-09 under pressure. |
| S-18 | Pedestrian step-model fallback (for the "hand the judge the phone" demo, clearly labelled) | **Should** | Yes | Required for the highest-impact 30 s of the demo — a *different, smaller model*, labelled as such on screen. |
| S-20 | Barometric floor / ramp detection | **Could** | Yes if ready | Demo-only, labelled. No IO-VNBD validation possible. |
| S-21 | Per-device IMU bias auto-calibration during stationary periods | **Could** | No | Cheap win, low demo value. |
| S-22 | Hindi UI and voice | **Could** | No | String architecture committed now; translation is Phase 2. |

**Everything else in the v3 PRD's §6.1 table (S-01, S-08, S-11, S-15 through S-17, S-19) belongs to Harshit's track — S-15/S-16's *heads* are his, but you consume their output in `NhcUpdate`/`AttitudeUpdate`; that consumption code is yours.** If your work seems to require training a model or writing a FastAPI route, that's a sign you're drifting into his scope.

---

## 4. Functional requirements you own (v3 PRD §8, full Given/When/Then carried over)

### Acquisition and conditioning

| ID | Requirement | Acceptance criteria |
|---|---|---|
| **FR-01** | Sensor acquisition at the highest rate the device supports. | **Given** an Android device with accelerometer and gyroscope, **when** the engine starts, **then** it registers at `SENSOR_DELAY_FASTEST`, records the *actual achieved* rate, and logs a warning below 50 Hz. |
| **FR-02** | High-rate pre-filtering of vibration and impulse noise. | **Given** a raw stream with injected impulse spikes, **when** pre-filtering runs, **then** spikes exceeding the Hampel criterion are replaced by the local median and residual energy above 5 Hz is attenuated by at least `[VERIFY: target dB, set after measuring real vibration spectra]`. |
| **FR-03** | Decimation to the model's operating rate with anti-aliasing. | **Given** a stream at rate `f`, **when** decimating to 10 Hz, **then** a low-pass with cutoff below 5 Hz is applied first and the output matches the Python reference on the golden vectors within tolerance. |
| **FR-33** | **Use hardware sensor timestamps, never wall clock, and resample onto a fixed grid.** | **Given** a batch of Android sensor events delivered irregularly, **when** the engine ingests them, **then** ordering and Δt come from `SensorEvent.timestamp`, out-of-order events are handled, and the resampled grid has no gap exceeding `[VERIFY]` ms. Tested with a recorded irregular-batch trace. *(This is the defect that silently kills projects like this. It is a requirement, not a note.)* |

### Alignment

| ID | Requirement | Acceptance criteria |
|---|---|---|
| **FR-04** | Estimate phone→vehicle roll and pitch from the gravity vector. | **Given** ≥3 s stationary or steady motion, **when** alignment runs, **then** roll and pitch are within `[VERIFY]`° of synthetic ground truth on the augmentation test set. |
| **FR-05** | Estimate the yaw offset between phone and vehicle forward axes. | **Given** ≥1 longitudinal acceleration event above threshold, **when** yaw estimation runs, **then** the estimate agrees with GNSS course-over-ground within `[VERIFY]`° while GNSS is available. |
| **FR-06** | Detect mid-drive remount and re-run alignment. | **Given** a synthetic rotation discontinuity at t=T, **when** the detector runs, **then** it flags within 2 s of T, re-initialises alignment, and inflates filter covariance accordingly. |

**This is the exact rotation stage §2.2 above requires happen before any window reaches Harshit's model — your alignment code is the boundary that makes "vehicle-frame-aligned" true.**

### Model consumption (shared ownership — you own the filter-side integration)

| ID | Requirement | Acceptance criteria | Your part |
|---|---|---|---|
| **FR-24** | Fall back safely when the model is unavailable or implausible. | **Given** a missing, corrupt or out-of-bounds model output, **when** inference is attempted, **then** the engine logs it, degrades to the NHC-only filter (B3 behaviour), sets a degraded mode pill, and does not crash. | You implement `ModelFallback.kt`, validating against physical bounds Harshit supplies (§9 open item). This is the enforcement half of a shared requirement — you write the check, he defines the numbers. |
| **FR-25** | Version and pin every model artefact. | **Given** a running engine, **when** a pose is emitted or a trace exported, **then** it carries the model version hash; loading a model whose hash is not in the signed manifest is refused. | You implement the load-time signature/hash check (`ModelManifest.kt`) against the public key Harshit's signing pipeline produced — **the public key is pinned in the app; the private key never reaches you, and you never need it to.** |

### The filter

| ID | Requirement | Acceptance criteria |
|---|---|---|
| **FR-09** | Propagate a 15-state error-state EKF (position, velocity, attitude, accel bias, gyro bias) at the device's native rate. | **Given** a stream of aligned samples, **when** the filter propagates, **then** state and covariance update without numerical failure over a 10-minute sequence and covariance remains positive-definite every step (Joseph-form or square-root update, asserted). |
| **FR-10** | Apply non-holonomic constraints as pseudo-measurements. | **Given** the vehicle is in motion, **when** the NHC update runs, **then** body-frame lateral and vertical velocity are driven toward zero with covariance from the context head or a fixed default, and the update is **suppressed** when stationary (FR-26) or reversing. |
| **FR-11** | Fuse predicted displacement weighted by its predicted variance. | **Given** a velocity-head output `(μ, σ²)`, **when** the update runs, **then** `R = σ² × trust_factor`, and a unit test confirms that doubling `σ²` halves the state correction magnitude. |
| **FR-26** | Apply a zero-velocity update when the vehicle is detected stationary. | **Given** IMU energy below the stationarity threshold and near-zero predicted displacement, **when** ZUPT runs, **then** velocity is driven to zero, **accelerometer and gyroscope bias states are re-observed**, NHC is suppressed, and the marker does not creep more than `[VERIFY]` m over a 120 s simulated idle. |
| **FR-27** | Gate every GNSS update with a chi-square test on the normalised innovation. | **Given** a fix and innovation covariance `S`, **when** the update is attempted, **then** `νᵀS⁻¹ν` is compared against the chi-square threshold for the measurement dimension at the configured confidence; a fix exceeding it is rejected and logged as a `MODE_EVENT` with trigger `innovation_gate`. A unit test injects a synthetic jump and asserts rejection. |
| **FR-32** | **Learned yaw-increment correction (Head D).** *Should.* | **Given** Head D is enabled, **when** it infers, **then** it returns a per-window yaw increment and variance fused as an attitude measurement; **and** the Week-5 ablation shows a heading-error improvement over Head-D-disabled at 5 seeds, otherwise the head is cut and the PRD records that it was. | You build `AttitudeUpdate.kt`, but **only wire it in once Harshit reports the Week-5 gate passed** — building it against a head that later gets cut wastes your time; wait for his signal. |

### Mode handover

| ID | Requirement | Acceptance criteria |
|---|---|---|
| **FR-12** | Monitor GNSS quality and classify each fix trusted / degraded / absent. | **Given** a fix with CN0 values, satellite count and reported accuracy, **when** the monitor runs, **then** it emits one of three states; a *degraded* fix is down-weighted or rejected per policy — **never silently accepted**. |
| **FR-13** | Continuous operation across GNSS loss with no output discontinuity. | **Given** GNSS stops at t=T, **when** the engine runs, **then** the pose stream contains no gap, no NaN and no position jump exceeding `[VERIFY]` m at T; the transition is logged. |
| **FR-14** | Smooth reacquisition, no visible teleport. | **Given** GNSS returns at t=T′ differing from the estimate, **when** the update is applied, **then** the correction is distributed over a configurable window (default 1 s) and the rendered marker's inter-frame displacement never exceeds a plausible vehicle speed. |
| **FR-34** | **RTS backward smoothing of the outage trace on reacquisition.** | **Given** an outage from T to T′, **when** GNSS is reacquired and accepted, **then** a Rauch–Tung–Striebel smoother is run over the buffered outage window and the *rendered trailing trace* is replaced by the smoothed path; **the live marker is never moved backwards in time.** Tested by asserting smoothed RMSE ≤ filtered RMSE over the outage on golden segments. |

### Map binding

| ID | Requirement | Acceptance criteria |
|---|---|---|
| **FR-15** | Offline map matching against a local OSM extract. | **Given** an `.osm.pbf` on device and a filtered trajectory, **when** the fixed-lag HMM matcher runs, **then** it returns a matched segment sequence and a per-fix confidence, **with no network access** — asserted by a no-socket test. |
| **FR-16** | **Refuse to snap when confidence is low.** | **Given** parallel candidate roads, **when** the posterior margin falls below threshold (default 0.20), **then** no map pseudo-measurement is applied, the indicator goes amber, and the raw filtered position is rendered. |
| **FR-28** | Bound matcher latency and prevent feedback lock-in. | **Given** a fixed-lag matcher with lag `L`, **when** a map pseudo-measurement is applied, **then** (a) reported pose lag never exceeds `L` (default 5 s) and the UI indicates a lagged match; (b) the matcher retains at least top-`k` hypotheses rather than committing; (c) **the map update can never reduce position covariance below a configured floor**, so a wrong snap cannot make the filter confident. |
| **FR-29** | **Road-manifold (1-D arc-length) constraint.** | **Given** the corridor detector finds exactly one candidate way within 3σ that is junction-free for the next `L_c` metres and whose posterior margin is decisive, **when** constrained mode engages, **then** (a) lateral offset from the polyline is driven to zero with tight covariance and heading is pulled to the local road bearing; (b) the covariance floor of FR-28 still applies; (c) the constraint **releases within one update** when a junction enters the horizon or the margin collapses; (d) engagement and release are logged as `MODE_EVENT`s and shown on the UI. Tested by `test_corridor_constraint_collapses_cross_track` and `test_corridor_disengages_at_junction`. |
| **FR-30** | **Magnetic route memory.** *Should.* | **Given** a stored magnetic signature for the current OSM way indexed by arc-length, **when** the vehicle traverses it under GNSS denial, **then** the live magnetometer trace is cross-correlated against the signature in 1-D and an along-track pseudo-measurement is applied **only if** the correlation peak exceeds the runner-up by a configured margin; otherwise no update is applied and the state is logged as `MAG_AMBIGUOUS`. Tested by `test_magnetic_relocalisation_on_repeat_route` using held-out IO-VNBD repeat-route pairs. |

### Interface, evidence, deployment

| ID | Requirement | Acceptance criteria |
|---|---|---|
| **FR-17** | Render a live map with marker, heading, confidence ring, mode pill and context label. | **Given** poses at ≥10 Hz, **when** the map view is visible, **then** the marker updates at ≥10 Hz, the ring radius equals the 95% horizontal uncertainty, and the pill reads `GNSS / FUSED / DR` with a `CORRIDOR` indicator when FR-29 is engaged. |
| **FR-18** | Replay a recorded CSV through the **identical engine code path** as live sensors. | **Given** an IO-VNBD CSV (converted to `contracts/replay_csv` schema), **when** replay runs, **then** the engine consumes it through the same `SensorSource` interface as the live device and produces a pose stream matching the reference within tolerance. |
| **FR-19** | Display live drift against ground truth in replay mode. | **Given** a replay with a ground-truth column, **when** it runs, **then** the UI shows instantaneous and cumulative horizontal error in metres and drift as a percentage of distance travelled. |
| **FR-20** | Export a trip trace for audit. | **Given** a completed trip, **when** the user exports, **then** a GeoJSON/CSV of timestamped poses, covariances, mode states and **the full provenance block** is written to device storage. |
| **FR-21** | Consume an external (non-phone) IMU stream at arbitrary rate. | **Given** a CSV or serial source at 200 Hz, **when** the edge CLI runs, **then** **the same compiled core** produces a pose stream, with the decimation stage adapting to the input rate and propagation running at 200 Hz. |
| **FR-22** | Operate with zero network access for the entire trip. | **Given** airplane mode with location disabled, **when** the engine runs in replay mode, **then** all functionality except live GNSS operates, and a build-level assertion fails CI if any positioning-path code opens a socket. |
| **FR-23** | Explicit, granular, withdrawable consent before any data leaves the device. | **Given** first launch, **when** the consent screen is shown, **then** telemetry defaults to **off**, each category's purpose is stated in plain language, and withdrawal is available at any time from settings **without degrading positioning**. |

> **Note on §8's own framing (v3 PRD):** most of your requirements are refusals — refuse to snap when unsure (FR-16), refuse a GNSS fix that contradicts felt motion (FR-27), refuse to let a map assertion create false certainty (FR-28), refuse to run an unsigned or contract-incompatible model (FR-25/`VERSIONING.md`). Treat every one of these with the same seriousness as a "must do" requirement — they're what makes the system degrade toward honesty instead of confident error (v3 PRD §14.8's governing principle).

---

## 5. Architecture — Tier 1 + Tier 2a, and why it's Kotlin (v3 PRD §10.0-10.5)

### 5.1 The three-tier statement, and your place in it

You own **Tier 1 (Presentation)** — the Android app's UI and the edge/CLI's terminal output — and **Tier 2a (on-device Application)** — the whole `core/` engine. Harshit owns Tier 2b (backend). **Tier 2a must run fully offline; it never assumes a network exists** (FR-22). The relationship to Tier 2b is one-directional and asynchronous: you pull a model or map extract when convenient, never mid-trip; you never block on the backend for anything in the positioning loop.

### 5.2 The dependency rule (v3 PRD §10.1, verbatim)

> "Source-code dependencies point inward only. Presentation depends on Application; Application depends on Data through interfaces it owns; no tier ever depends on a tier above it. The Presentation tier contains no business rule and never touches a repository. The Data tier contains no business rule and never calls a service."

**Enforced mechanically, not by convention:** a Gradle module-dependency check plus an ArchUnit test (`TierDependencyTest`) fails the build if `:android` types leak into `:core`, or if `:core` imports anything Android-specific. Set this up early — it's cheap now and expensive to retrofit once violations have crept in.

### 5.3 One core, three consumers — why Kotlin, not C++ (v3 PRD §10.2)

**One Gradle module (`core/`) compiles to both an Android library and a plain `.jar`.** This is the entire mechanism behind "the desktop number is evidence for the phone" and behind FR-21's edge deployment being nearly free — a phone `SensorSource`, a CSV replay `SensorSource`, and a serial-IMU `SensorSource` are three implementations of the same interface feeding the same compiled engine. **Build `core/` with zero Android-specific dependencies from day one** — this is what makes the `.jar` target actually work, not just theoretically possible.

**Performance headroom, so you know the budget isn't tight:** the 15×15 ESKF propagate is ~3,400 multiply-adds. On preallocated primitive `DoubleArray`s with **zero allocation in the hot path**, the JVM does this in single-digit microseconds — against a 5,000 µs budget per sample at the 200 Hz edge rate, that's roughly two orders of magnitude of headroom. **The residual risk is GC pause, not throughput** — mitigated by the zero-allocation rule (never allocate inside `ErrorStateEkf.propagate()` or any per-tick hot path; preallocate all working arrays once at startup) and asserted by a **30-minute soak test on p99.9 propagation latency**, which you build and run starting Week 3 (v3 PRD R-06).

### 5.4 Repo layout you own (v3 PRD §10.5, filtered)

```
core/                      :core — Kotlin. THE ONLY PLACE BUSINESS RULES LIVE.
└── src/main/kotlin/org/anchor/
    ├── sensors/            SensorSource + Phone/CsvReplay/SerialImu implementations
    ├── prefilter/          Hampel, Notch, Decimator, HardwareClockResampler (FR-33)
    ├── alignment/          GravityAlign, YawResolver, RemountDetector
    ├── model/               ModelRunner (ONNX Runtime Mobile), ProvenanceGuard, ModelManifest
    ├── fusion/              ErrorStateEkf, NhcUpdate, ZuptUpdate, VelocityUpdate,
    │                        ChiSquareGate, RtsSmoother, StationarityDetector
    ├── corridor/            CorridorDetector, ManifoldConstraint          (FR-29)
    ├── magnetic/            SignatureBuilder, ArcLengthCorrelator         (FR-30)
    ├── mapmatch/            FixedLagViterbi, ConfidenceGate, CovarianceFloor, OsmIndex
    ├── mode/                GnssQualityMonitor, ModeManager, ReacquisitionSmoother
    ├── math/                Mat, Vec — hand-written, zero-allocation, no dependency
    └── orchestrator/        EngineOrchestrator, EngineScheduler, TripExporter
└── src/test/kotlin/         One named test file per FR (see §11's traceability matrix below)

android/                    Presentation tier. NO business rules.
└── app/src/main/           Compose UI, ViewModels, DI wiring

edge/                       CLI engine — CSV/serial in, pose stream out    (FR-21)
```

**Note:** `reference/` (the Python oracle — a NumPy mirror of `core/`, used to generate golden vectors your filter must reproduce) is Harshit's to maintain, but you're the one who actually needs it — treat its golden vectors as your regression fixtures.

### 5.5 Traceability matrix — your file, exactly (v3 PRD §11, filtered to your FRs)

| FR | Files | Test |
|---|---|---|
| FR-01 | `core/…/sensors/AndroidSensorSource.kt` | `SensorRateReportingTest.kt` |
| FR-02 | `core/…/prefilter/Hampel.kt`, `Notch.kt` | `PrefilterImpulseRejectionTest.kt` |
| FR-03 | `core/…/prefilter/Decimator.kt` | `DecimatorGoldenVectorTest.kt` |
| FR-04 | `core/…/alignment/GravityAlign.kt` | `GravityRollPitchTest.kt` |
| FR-05 | `core/…/alignment/YawResolver.kt` | `YawOffsetVsGnssCourseTest.kt` |
| FR-06 | `core/…/alignment/RemountDetector.kt` | `RemountDetectionLatencyTest.kt` |
| FR-09 | `core/…/fusion/ErrorStateEkf.kt`, `math/Mat.kt` | `EkfCovariancePsdLongRunTest.kt`, `EkfZeroAllocationSoakTest.kt` |
| FR-10 | `core/…/fusion/NhcUpdate.kt` | `NhcSuppressedWhenStationaryTest.kt` |
| FR-11 | `core/…/fusion/VelocityUpdate.kt` | `VelocityUpdateScalesWithVarianceTest.kt` |
| FR-12 | `core/…/mode/GnssQualityMonitor.kt` | `GnssClassificationStatesTest.kt` |
| FR-13 | `core/…/mode/ModeManager.kt`, `orchestrator/EngineOrchestrator.kt` | `NoOutputGapAcrossOutageTest.kt` |
| FR-14 | `core/…/mode/ReacquisitionSmoother.kt` | `ReacquisitionNoTeleportTest.kt` |
| FR-15 | `core/…/mapmatch/FixedLagViterbi.kt`, `OsmIndex.kt` | `MapmatchOfflineNoNetworkTest.kt` |
| FR-16 | `core/…/mapmatch/ConfidenceGate.kt` | `RefuseSnapOnAmbiguousCandidatesTest.kt` |
| FR-17 | `android/…/MapScreen.kt`, `MapViewModel.kt` | `MapViewModelStateTest.kt` |
| FR-18 | `core/…/sensors/CsvReplaySource.kt` | `ReplayMatchesReferenceTest.kt` |
| FR-19 | `android/…/DriftPanel.kt`, `DriftViewModel.kt` | `DriftComputationTest.kt` |
| FR-20 | `core/…/orchestrator/TripExporter.kt` | `TripExportSchemaTest.kt` |
| FR-21 | `edge/src/main/kotlin/Main.kt`, `core/…/sensors/SerialImuSource.kt` | `Edge200HzReplayTest.kt` |
| FR-22 | build assertion in CI | `NoSocketInPositioningPathTest.kt` |
| FR-23 | `android/…/ConsentScreen.kt`, `core/…/orchestrator/ConsentGate.kt` | `ConsentDefaultsOffTest.kt` |
| FR-24 | `core/…/model/ModelFallback.kt` | `DegradeToNhcOnModelFailureTest.kt` |
| FR-25 | `core/…/model/ModelManifest.kt` | `RejectUnsignedModelTest.kt` |
| FR-26 | `core/…/fusion/ZuptUpdate.kt`, `StationarityDetector.kt` | `ZuptNoCreepWhenIdleTest.kt` |
| FR-27 | `core/…/fusion/ChiSquareGate.kt` | `ChiSquareRejectsPositionJumpTest.kt` |
| FR-28 | `core/…/mapmatch/FixedLagViterbi.kt`, `CovarianceFloor.kt` | `MapCannotDriveCovarianceBelowFloorTest.kt`, `MatcherLagBoundTest.kt` |
| **FR-29** | `core/…/corridor/CorridorDetector.kt`, `ManifoldConstraint.kt` | `CorridorConstraintCollapsesCrossTrackTest.kt`, `CorridorDisengagesAtJunctionTest.kt` |
| **FR-30** | `core/…/magnetic/SignatureBuilder.kt`, `ArcLengthCorrelator.kt` | `MagneticRelocalisationOnRepeatRouteTest.kt`, `MagRefusesOnWeakMarginTest.kt` |
| **FR-32** | `core/…/fusion/AttitudeUpdate.kt` | (shared with ML — wire only once Harshit's Week-5 gate passes) |
| **FR-33** | `core/…/prefilter/HardwareClockResampler.kt` | `HardwareTimestampResamplingTest.kt` |
| **FR-34** | `core/…/fusion/RtsSmoother.kt` | `RtsSmoothedRmseNotWorseTest.kt` |
| — | `core/build.gradle.kts`, `android/app/build.gradle.kts` | `TierDependencyTest.kt` |

---

## 6. Engine internals worth carrying in full (v3 PRD §9, §10.4, §14.8, §15 filtered)

### 6.1 Latency budget (v3 PRD §9.1) — your performance contract

| Stage | Runs at | p95 | Amortised per 100 ms tick |
|---|---|---|---|
| Sensor callback → pre-filter | ~100-200 Hz | 0.05 ms/sample | ~1.0 ms |
| Decimation + feature assembly | 10 Hz | 2 ms | 2.0 ms |
| **ANCHOR-Net inference** (Harshit's artifact, your `ModelRunner`) | 10 Hz | ≤15 ms | 15.0 ms |
| ESKF propagate + all updates | ~100-200 Hz | 0.03 ms/step | ~0.6 ms |
| Road-manifold constraint | 10 Hz | 0.3 ms | 0.3 ms |
| Map matching (fixed-lag HMM step) | 2 Hz | 10 ms | 2.0 ms |
| Magnetic correlation | 1 Hz, corridor only | 4 ms | 0.4 ms |
| **Engine total per 100 ms tick** | — | — | **≈ 21 ms p95, leaving ≈ 79 ms headroom** |
| Pose → rendered frame | 60 Hz | 16 ms | — |
| **User-visible: motion → marker moves** | — | **≤ 55 ms p95** | — |
| Edge engine, 200 Hz IMU | 200 Hz | ≤0.2 ms/sample | vs a 5,000 µs budget |

`[VERIFY: benchmark on a 4 GB mid-range device — Ravi's phone, not a flagship. Week-6 gate.]`

**Degradation policy when the budget is exceeded** (`EngineScheduler`, tested under simulated CPU starvation), in order:
1. Filter propagation is **never skipped** — cheapest and most critical.
2. **Model inference is skipped first** — a missed window is an increased-covariance gap, not an error.
3. Magnetic correlation second.
4. Map matching third.
5. Rendering throttled last.

### 6.2 Failure modes and fallbacks you own (v3 PRD §14.8, filtered)

| Failure mode | Detection | Fallback |
|---|---|---|
| Model missing, corrupt or unsigned | Manifest hash check at load (FR-25) | FR-24: degrade to NHC-only (B3), degraded pill, log |
| Model returns a physically impossible value | Bounds validator (against bounds Harshit supplies) | Reject the measurement; treat as a gap; inflate covariance |
| Sustained low confidence (σ² high >10 s) | Threshold on predicted variance | Widen the ring aggressively; beyond a limit, say "position uncertain" rather than showing a precise-looking marker |
| Vehicle stationary, engine idling | Head C reports `idle`; zero-velocity detector corroborates | ZUPT (FR-26). NHC suppressed. Bias states re-observed for free. |
| **Vehicle reversing** | No gear signal on a phone; detect via integrated longitudinal-acceleration sign + map context | **Known limitation (R-08).** Named, not hidden. |
| Map matcher snaps to the wrong road | Confidence gate (FR-16) | **Refuse to snap.** A drifting-but-honest position beats a confident lie. |
| **Corridor constraint engages on the wrong corridor** | Candidate count, junction horizon, posterior margin (FR-29) | Engage only when unambiguous; release within one update; FR-28's covariance floor still applies — the constraint can never make the filter certain. |
| **Magnetic match locks onto the wrong peak** | Correlation peak margin vs runner-up (FR-30) | **Refuse to correct.** Log `MAG_AMBIGUOUS`. |
| Phone picked up mid-drive | Remount detector (FR-06) | Inflate covariance, re-align, suppress velocity updates until convergence |
| GNSS returns with a multipath fix | Chi-square gate (FR-27) | Reject or heavily down-weight |
| Gyro yaw bias accumulates over a long outage | Heading covariance growth; disagreement with map road bearing | Bias re-observed at every ZUPT; road bearing as heading constraint; road-manifold constraint removes the consequence entirely inside a corridor; **magnetometer deliberately not trusted as a compass** |
| **Slow, sophisticated GNSS spoofing** | **Not reliably detectable by the chi-square gate** | Named limitation, not a solved problem — FR-31 (Harshit's bench) measures exactly where this boundary sits |

**The governing principle across every row: the system degrades toward honesty, never toward confident error. Every fallback ends in either a wider stated uncertainty or an explicit refusal. None ends in a crash or a silently wrong number.**

### 6.3 Security items you implement (v3 PRD §15, filtered)

| # | Threat | Your mitigation |
|---|---|---|
| **T1** | Malicious model substitution | FR-25: hash-check + signature verify at load, public key pinned in the app; refuse and fall back to B3 on failure |
| **T3** | GNSS spoofing | Chi-square gate (FR-27) rejects fixes disagreeing with felt motion. **State precisely: "detects discontinuous spoofing and multipath," never "spoof-proof."** A patient slow-drag spoof stays inside the gate the whole way. |
| **T4** | Trip-trace exfiltration from the device | Database encrypted at rest (SQLCipher / Android `EncryptedFile`), key in Android Keystore hardware-backed where available. 30-day retention. **Raw IMU never persisted at all — memory-only ring buffer, overwritten continuously.** |

---

## 7. Integration & Failure-Mode Playbook

This is the comprehensive enumeration the task calls for. Every row names a concrete, standard-practice fix — not an exotic invention.

### 7.1 Dependency / environment drift

| Failure mode | Why it happens | Fix |
|---|---|---|
| Gradle/AGP/Kotlin compatibility matrix issues — opaque sync failures | AGP, Gradle, and Kotlin each have their own supported-pairing matrix; a hand-picked combination outside it fails Gradle sync with a confusing error, not a clear version-mismatch message | Pick versions from **Android Studio's own bundled recommendation**, and record them in `TOOLCHAIN.md`'s Kotlin/Android table (currently blank placeholders — fill this in together with Harshit in your very first working session). Don't hand-override unless you both explicitly agree and re-record. |
| No Gradle lockfile — "works on my machine" | Floating `+`/`latest.release` dependency versions | Use Gradle's [version catalog](https://docs.gradle.org/current/userguide/platforms.html) (`gradle/libs.versions.toml`) with exact pinned versions for every dependency. Commit it. |
| Locally-installed Gradle vs. the wrapper | Someone runs a system `gradle` instead of `./gradlew` | **Always use the Gradle wrapper.** Commit `gradle/wrapper/gradle-wrapper.properties`. Never rely on a locally-installed Gradle version — that is exactly the "works on my machine" trap `TOOLCHAIN.md` warns against. |
| `onnxruntime-android` AAR version unpinned or supports a different opset than the model needs | Adding the dependency with a `+` range, or picking a version without checking its release notes against the model's opset | **Pin an exact version in `build.gradle.kts`, never a `+` or range.** Before picking it, check the AAR's release notes for minimum supported ONNX opset. Re-check this every time Harshit's model architecture changes (§2.2) — it's the row `TOOLCHAIN.md` calls out as "the row most likely to cause a runs-in-Python-crashes-on-Android bug if ignored." |
| JDK version mismatch | Different Android Studio installs bundle different JDKs | Record the exact version `java -version` reports in `TOOLCHAIN.md`, together, once. |

### 7.2 Model I/O drift

| Failure mode | Why it happens | Fix |
|---|---|---|
| Window size or feature order silently wrong on your side | You hand-write the windowing/feature-assembly code independently of the contract, and it drifts from `model_manifest.json`'s actual values | **Read `window_size_samples`, `feature_order`, `input_shape` from the manifest at build/runtime, don't hardcode `20` and the six feature names as magic literals scattered through `ModelRunner.kt` and `Decimator.kt`.** Where you must hardcode (e.g. for performance in a hot path), keep the literal directly next to a comment citing the manifest field it must match, and cover it with a golden-vector test that would fail if it drifted. |
| Normalization stats hardcoded and drift from Harshit's real values | Someone pastes a mean/std array into Kotlin during a debugging session "just to unblock testing" and it's never removed | **There is exactly one legitimate source: read `model_manifest.json`'s `normalization.mean`/`normalization.std` (or the ONNX file's embedded metadata) at model-load time.** Never a Kotlin-side literal array. Build this reading path in Week 1-2 even while the stub's stats are a no-op (`[0,0,0,0,0,0]`/`[1,1,1,1,1,1]`) — the no-op will not stay a no-op once Harshit ships real training stats, and you don't want to be retrofitting this under Week-6 pressure. |
| Mean-vs-log-variance-vs-std confusion | Treating `velocity_log_variance` as if it were already variance, or std-dev | **Always `exp()` it before use, in exactly one function** (e.g. `ModelOutput.variance()`), never inline `exp()` calls scattered through `VelocityUpdate.kt`. If Harshit's contract ever changes this parametrization, that's a MAJOR version bump you'll hear about before it ships — but structure your code so the conversion lives in one place regardless. |
| Batch-dimension mismatches | Forgetting to wrap your `[20][6]` window into a `[1,20,6]` tensor, or vice versa when reading output `[1,1]` back to a scalar | The contract is explicit: input `[1, 20, 6]`, outputs `[1, 1]`. Write a unit test that constructs a known window, runs it through `ModelRunner`, and checks the returned scalar types — this is cheap and catches shape bugs immediately rather than as a runtime ONNX Runtime exception mid-demo. |
| Quantization changes numeric behavior | Harshit ships an int8-quantized model whose golden vectors need a looser tolerance than the fp32 stub's `1e-4` | Your instrumented golden-vector test (§2.3, the one you're building) should read `tolerance_abs` from each vector file rather than hardcoding `1e-4` — so a legitimate tolerance change on his side doesn't require you to also patch your test's assertion logic. |

### 7.3 Units and data mismatches

| Failure mode | Why it happens | Fix |
|---|---|---|
| Gyro deg/s vs rad/s | Android's `TYPE_GYROSCOPE` is already rad/s (per `contracts/units.md`) — the risk here is a *replay CSV* someone produced from a different source (e.g. a converted IO-VNBD file) carrying the wrong unit | Assert `abs(value) < 10 rad/s` (~573°/s — no road vehicle yaws that fast) on **every gyro sample your engine ingests, live or replayed**, at the point of ingestion. `validate_replay_csv.py` does the CSV-side version of this check; carry the equivalent assertion in Kotlin sensor-ingestion code, per `units.md`'s explicit instruction. |
| km/h vs m/s | A GNSS provider or an external serial IMU reports speed in km/h | Convert at the boundary (÷3.6), immediately, in the `SensorSource` implementation that ingests it — never carry km/h into `core/`'s internal math. |
| Degrees vs radians | Heading math mixing UI-facing degrees with filter-internal radians | `units.md`'s rule: radians in all filter math, degrees only at the UI/OSM boundary, through **one single named conversion function** — not scattered `* 180/pi` calls. Audit `MapViewModel.kt` and anywhere OSM bearings are consumed for this. |
| **Coordinate frame / axis convention — phone frame vs vehicle frame vs NED/ENU** | `contracts/units.md` fixes unit *magnitudes* but explicitly does not fix which axis is forward, which is up, or right-vs-left-handedness. This is a real gap you must close, not something the scaffold already resolved. | **This is an open decision you must lock down with Harshit in Week 1 — see §9.** Your `AlignmentService` (FR-04/FR-05) is the *only* place this convention gets applied; document it as a code comment at the top of `GravityAlign.kt` and `YawResolver.kt` referencing whatever `contracts/frame_convention.md` you and Harshit agree to write, and make sure your `core/` internal state representation (position/velocity/attitude in the ESKF) and Harshit's training-label sign convention agree. A mismatch here won't crash — it will make the model's velocity estimate consistently biased in sign for lateral motion or turns, which is exactly the kind of bug that looks like "the model just isn't very good" and wastes days debugging the wrong layer. |
| Timestamp epoch/unit confusion | Mixing `SensorEvent.timestamp` (nanoseconds since an arbitrary boot-relative epoch, **not** Unix epoch) with wall-clock milliseconds | FR-33 exists precisely for this: use hardware sensor timestamps for ordering/Δt, but **when writing to the replay-CSV or exported-trace schema's `timestamp_ms` field (milliseconds since Unix epoch UTC), you must convert deliberately** — `SensorEvent.timestamp` alone is not directly usable as `timestamp_ms`; anchor it against a wall-clock reference sample taken once at stream start. Get this wrong and every exported trip trace has a systematically wrong absolute time, even though internal Δt-based math still works — a subtle bug that only shows up when someone tries to correlate a trace against something external. |
| Missing-value sentinel bugs (0.0 vs empty) | Your `CsvReplaySource` writes `0.0` for a GNSS-invalid row instead of leaving the field empty | Never do this. `schema.json`'s explicit rule: `0,0` is a real coordinate (off West Africa) and will silently corrupt map matching rather than erroring. Any code you write that *produces* a replay CSV (test fixtures, exported trip traces in the same schema) must follow the same empty-string convention as `make_sample_csv.py`. |
| Locale decimal-separator corruption | A CSV gets opened/re-saved in Excel on a non-US-locale Windows machine | Never open/re-save any contract CSV in Excel. Edit with a script or plain text editor. `validate_replay_csv.py` catches a stray comma inside a numeric field — run it on any CSV you're about to commit or hand to Harshit. |
| CRLF/BOM encoding issues | Windows text editors default to CRLF; some add a UTF-8 BOM | `.gitattributes` forces LF on checkout (already in the scaffold). **A naive Kotlin `BufferedReader().readLine()` split on `,` will silently include a trailing `\r` in the last field of every row if a CRLF file ever slips through** — write your `CsvReplaySource`'s line-splitting to explicitly trim `\r`, and validate any CSV you produce with `validate_replay_csv.py` before it enters the repo. |

### 7.4 API/contract drift

| Failure mode | Why it happens | Fix |
|---|---|---|
| snake_case vs camelCase field mismatch | A Kotlin data class hand-written against a guessed field name (`download_url`) instead of the actual wire format (`downloadUrl`) — deserializes to null silently, with no exception | **Solved structurally on Harshit's side by `CamelModel`** — every response is genuinely camelCase on the wire. **Your job: match it exactly.** Prefer codegenning your Kotlin client from `openapi.json` (e.g. via an OpenAPI Generator Gradle plugin) over hand-writing data classes, so a contract change surfaces as a regenerate-and-diff rather than a silent runtime null. If you do hand-write, review every field name against the actual `openapi.json`, not against your memory of the shape. |
| Enum value mismatches | Your Kotlin `enum class MapRegion` diverges from Python's `MapRegion` (e.g. you add a region locally for testing and forget it's not on the server) | Treat `MapRegion` and any other server-defined enum as generated/derived from `openapi.json`, not hand-maintained independently. If you need a region that doesn't exist server-side, that's a message to Harshit and an `API_CONTRACT_VERSION` bump, not a local addition. |
| Optional/required field disagreements | Your client assumes a field is always present; the server's Pydantic model marks it optional | Check nullability explicitly against `openapi.json`'s schema for every field you deserialize — don't assume non-null from a stub response that happens to always populate it. |
| Assuming an endpoint from v3 PRD §13 exists when it isn't in the current `openapi.json` | The v3 PRD describes a fuller API surface (`/fleet/trips`, `/magsig/batch`, etc.) than what's actually implemented today | **Before writing a Kotlin client method, check the actual `openapi.json` for that endpoint.** If it's not there, it doesn't exist yet — confirm with Harshit whether/when it will, rather than building against an assumed shape that may differ from what he eventually ships. |
| No agreed error-response envelope | Different endpoints returning differently-shaped error bodies as they're added ad hoc | This is flagged as an open item (§9) — don't build bespoke per-endpoint error parsing until the shape is agreed; a generic "parse whatever FastAPI's default `HTTPException` body looks like today" will break the moment Harshit implements the structured envelope v3 PRD §13 describes. |

### 7.5 Versioning/release management

| Failure mode | Why it happens | Fix |
|---|---|---|
| No compatibility matrix | Nobody tracks which app build needs which model/API contract version | `VERSIONING.md`'s compatibility matrix table exists for exactly this — read it before assuming a contract version is safe, and flag to Harshit if you notice it's stale relative to what's actually shipping. |
| **App silently tries to load an incompatible model instead of refusing** | The refusal check either doesn't exist, or exists but isn't called on every model-load path (first launch vs. OTA update vs. replay harness swap) | **Implement `MIN_SUPPORTED_CONTRACT_VERSION` as a single named constant, checked in exactly one function that every model-load code path calls** (§2.6). This is the concrete mechanism behind FR-24's fallback and the `VERSIONING.md` refusal rule — a model that fails the check must never run "just this once to see." Write a test (`RejectUnsignedModelTest.kt` covers the signature half; add the equivalent for the contract-version half) that constructs a deliberately-incompatible manifest and asserts the app refuses and falls back rather than crashing or silently proceeding. |
| No changelog discipline on contract changes | A model version bumps with no record of why | Not primarily your discipline to enforce (Harshit's side generates the model), but **your app should log the model version hash into every exported trip trace (FR-25)** — this is the passive changelog that lets you correlate a bad demo run against exactly which model was loaded, after the fact. |

### 7.6 Git/process risks

| Failure mode | Why it happens | Fix |
|---|---|---|
| Two people editing the same file | Both touch `contracts/` occasionally | `contracts/` is the one shared folder — touch it only with a heads-up message first, per `VERSIONING.md`. `core/`, `android/`, `edge/` vs. `ml/`, `backend/` are exclusively owned per track, so this shouldn't come up outside `contracts/`. |
| Merge conflicts from long-lived branches | Working in isolation for a week before merging | Branch-per-feature, short-lived, **recommend merging to `main` at least every other day**, more often during integration weeks (P2 onward, per v3 PRD §20's roadmap). |
| Unreviewed pushes to `main` | No branch protection configured | Set this up in your very first working session (step 3 of `README.md`'s shared setup checklist): protect `main`, require PR review + the `contracts-ci` status check before merge. |
| Secrets accidentally committed | An API key or signing artifact pasted during debugging | `.env.example` with variable names, no values, committed; real `.env` gitignored from commit one. Install `gitleaks` pre-commit + CI hook in Week 1, before there's history to clean. **You never need the model-signing private key — only the public key, pinned in the app** (v3 PRD §15.3); if you ever find yourself asking Harshit for the private key, that's a sign of a design mistake, stop and reconsider. |
| Large binaries bloating the repo | An `.osm.pbf` extract or a `.apk`/`.aab` build artifact gets committed | `.gitignore` already excludes `.gradle/`, `/build/`, `*/build/`, `*.osm.pbf`, `local.properties`, `.idea/`, `*.iml`, `captures/`, `.cxx/`. Real map extracts and model weights ship through the backend's registry/CDN, never committed — only the small stub artifacts under `contracts/` are ever committed binaries. |

### 7.7 CI/testing discipline

| What | Command | When | Owner |
|---|---|---|---|
| **Android instrumented golden-vector test** | *Not yet built — this is your primary to-build item from the CI file.* `./gradlew :app:connectedContractCheckDebugAndroidTest` (once you build it) | Before trusting any new `.onnx` from Harshit, and in CI on every push | **You.** Uncomment `android-contract-check` in `.github/workflows/contracts-ci.yml` once it exists. |
| Replay CSV schema regeneration + validation | `python make_sample_csv.py` then `python validate_replay_csv.py sample_replay.csv` (or your Kotlin-side equivalent parser test) | Before every push touching `contracts/replay_csv/` | You |
| `TierDependencyTest` (ArchUnit) | Part of `core`'s Gradle test task | Every build | You |
| `EkfZeroAllocationSoakTest` (30-min p99.9 GC soak) | Scheduled/manual — too long for every push | From Week 3 onward, before any Week-6-gate claim about latency | You |
| No-socket assertion in positioning path | Build-level assertion in CI (FR-22) | Every push | You |
| `gitleaks` secret scan | Pre-commit hook + CI | Every commit, from Week 1 | Shared setup |

---

## 8. Week-by-week checklist

*Consistent with v3 PRD §20's phasing (P0-P4), scoped to your deliverables. Explicit sync points with Harshit are marked **SYNC**.*

### Week 1 (P0 — Foundations)

- [ ] **Day 1:** resolve R-02 (dataset vehicle count) alongside Harshit if needed for split-protocol wording — mostly his task, but confirm you're not building any "held out by vehicle" assumption into your own tooling that the data can't support.
- [ ] Map-matching lag ADR (R-13) — **decide fixed-lag Viterbi vs. forward-only now**, in P0, and record it as an ADR. Default per the v3 PRD: fixed-lag Viterbi, `L=5s`, UI marks the lagged segment; the filter's own unlagged pose renders the live marker, the map correction applies retroactively to the trailing trace (same mechanism as FR-34).
- [ ] **SYNC:** fill in `TOOLCHAIN.md`'s Kotlin/AGP/Gradle/onnxruntime-android rows together with Harshit — five minutes, prevents a week of "works on my machine."
- [ ] **SYNC:** lock down the phone-frame/vehicle-frame/NED-ENU axis convention (§7.3, §9) — write it into a new `contracts/frame_convention.md`.
- [ ] Python reference stack (Harshit maintains `reference/`, but confirm early what B2/B3 outputs you'll be testing your Kotlin filter against).
- [ ] **Stub model wired in from day one (R-15)** — build your full pipeline against `anchor_net_stub.onnx` immediately, even though it's a meaningless linear stub. This is what lets the whole engine run end-to-end from Week 2 instead of a big-bang integration in Week 7.
- [ ] Start the replay harness (S-06) — **this is the single most load-bearing component in the demo and it's Must, P0, before the UI.**
- [ ] Repo hygiene: `gitleaks` pre-commit hook, `TierDependencyTest` skeleton, branch protection on `main` requiring PR review + `contracts-ci`.
- [ ] `CsvReplaySource` parsing `contracts/replay_csv/schema.json`'s exact 15-column schema.

### Week 2 (P0 continued)

- [ ] `SensorSource` interface finalized; `AndroidSensorSource`, `CsvReplaySource` implementations working against the stub model end to end.
- [ ] Pre-filter stack (Hampel, notch, decimator) with FR-33's hardware-timestamp resampling.
- [ ] Alignment (`GravityAlign`, `YawResolver`) — this is the boundary that makes vehicle-frame-aligned inputs true for the model contract (§2.2).
- [ ] Strapdown INS (B2) and ESKF + NHC + ZUPT + chi-square gate (B3) in Kotlin, tested against the Python reference on golden vectors.
- [ ] **Exit criterion for P0 (v3 PRD §20): B2 and B3 produce plots on held-out data, and the full pipeline runs end to end against the stub model.**

### Weeks 3-5 (P1 — engine work in parallel with Harshit's model work)

- [ ] Kotlin core port of the full fusion stack, in parallel from Week 3 — it depends only on the stub interface, not on real weights.
- [ ] Build the **Android instrumented golden-vector test** (§2.3, §7.7) — this is your to-build item flagged from the CI placeholder; don't let it slip to the end.
- [ ] `VelocityUpdate.kt` wired against the stub model's output, with the `exp(log_variance)` conversion in exactly one place.
- [ ] `ModelManifest.kt`'s `MIN_SUPPORTED_CONTRACT_VERSION` refusal check implemented and tested against a deliberately-incompatible manifest.
- [ ] **Week-5 gate: wait for Harshit's report** (v3 PRD §20.1) — if the velocity head beats B3 by ≥20% relative, proceed as planned; if it beats B3 but weakly, or fails calibration, expect a message that Head C/D are cut and `VelocityUpdate` falls back to a fixed `R`; if it doesn't beat B3 at all, the demo pivots to B3 as the shipped system and **FR-29 (your road-manifold constraint) and FR-31 (Harshit's integrity bench) become the headline capabilities** — both independent of the ML, which is exactly why they were chosen as differentiators in the first place.
- [ ] **SYNC:** the moment Harshit ships a real-weights `.onnx` (replacing the stub), run your instrumented golden-vector test against it immediately, before wiring it into anything you'll demo from.

### Weeks 6-7 (P2 — On-device)

- [ ] **Week-6 gate: on-device latency benchmark + GC soak, on a mid-range device (Ravi's phone, not a flagship).** Coordinate with Harshit on getting his latest exported model onto your test device.
- [ ] `ModelRunner.kt` via ONNX Runtime Mobile, reading normalization stats from the manifest (not hardcoded).
- [ ] Android app: map, marker, ring, pill, drift panel, context label.
- [ ] **FR-29 road-manifold constraint** — your primary differentiated engineering effort this phase.
- [ ] FR-34 RTS smoother.
- [ ] Coordinate with Harshit on FR-31's integrity bench — his attack injector needs your `ChiSquareGate.kt` to test against; make sure it's stable before he schedules the bench run.

### Week 8 (P3 — Binding and polish)

- [ ] OSM extracts for the demo corridors — **Week-4 audit should already have flagged geometry quality issues (R-07); resolve before locking the corridor.**
- [ ] Fixed-lag HMM + confidence gate + covariance floor finalized (FR-15, FR-16, FR-28).
- [ ] Mode handover (FR-12, FR-13, FR-14) polished for the demo's split-screen moment.
- [ ] **FR-30 magnetic route memory** (Should) — if it's not working cleanly by here, it's the first thing cut per S-14's note; report a clean negative result with correlation statistics if it doesn't pan out (v3 PRD R-19 — this is a legitimate, even good, outcome to report honestly).
- [ ] Edge CLI (`edge/`) — one terminal screenshot with a 200 Hz stream, discharging the PS's dual-target requirement.
- [ ] Pedestrian fallback model integration + its honesty labelling (R-10 — the UI must say "pedestrian mode — different model" and the presenter says so out loud before a judge starts walking, non-negotiable).

### Final week (P4 — Freeze and evidence)

- [ ] Full demo script rehearsed end to end, twice, with every fallback exercised (v3 PRD §19.1's fallback table).
- [ ] Corridor video recorded at least one week before travel.
- [ ] Results sheet with provenance hashes printed.
- [ ] Confirm the app's `MIN_SUPPORTED_CONTRACT_VERSION` and Harshit's shipping model/API versions are actually compatible — check `VERSIONING.md`'s compatibility matrix one last time before freeze.

---

## 9. Open decisions — flagged, not silently resolved, must be settled with Harshit in Week 1

1. **Axis / frame convention.** `contracts/units.md` fixes unit magnitudes but not the coordinate frame — which axis is forward, up-vs-down sign, right-handed vs. left-handed, NED vs. ENU for any north-referenced quantity, and the exact rotation your `AlignmentService` applies from phone frame to vehicle frame. **Recommended default: right-handed vehicle frame, x=forward, y=left, z=up (ISO 8855-style), confirmed against IO-VNBD's own labelled sign conventions before Harshit's real training starts.** Write the agreed convention into a new `contracts/frame_convention.md`, reference it from `model_manifest.json`, and document it directly in `GravityAlign.kt`/`YawResolver.kt`'s code comments.
2. **`velocity_mean_mps` vs. displacement semantics.** The scaffold's current model output is named/described as mean *speed*; v3 PRD §14.1 specifies the actual training target as scalar *displacement* over the window. Your `VelocityUpdate.kt` measurement model differs depending on which it is — get an explicit answer from Harshit before wiring the real model in, and structure the conversion as a single named function so a later resolution is a one-line change, not a scattered refactor.
3. **The error-response envelope shape.** Not yet implemented consistently in `stub_api.py`. Agree on the real shape before you build generic error-handling/retry logic in your network client.
4. **The fuller §13 API surface vs. the current 4-endpoint stub.** Decide with Harshit which of the v3 PRD's roadmap endpoints (fleet queries, magnetic-signature upload, device enrolment) are actually needed for the hackathon demo, so you're not assuming an endpoint exists that hasn't been built.
5. **Model bounds for FR-24's implausibility check.** Harshit owns the numbers; get them documented somewhere you can read programmatically (extended `model_manifest.json` or a sibling file) rather than communicated informally, so `ModelFallback.kt`'s bounds don't silently drift from what he actually intends.

---

## 10. What "done" looks like before you push, every time

Before any push touching `contracts/replay_csv/`, or anything in `core/` that consumes `contracts/model_io/` or `contracts/backend_api/`:

1. Did you read the actual contract file (`model_manifest.json`, `schema.json`, `openapi.json`) rather than working from memory of what it used to say?
2. If you touched `contracts/replay_csv/`: did you change the generator, not `sample_replay.csv` directly, and re-run it?
3. Does `validate_replay_csv.py` pass on your regenerated sample?
4. If you're about to trust a new `.onnx` file: has it passed your instrumented golden-vector test?
5. Is `MIN_SUPPORTED_CONTRACT_VERSION` still correctly enforced, and did you test the refusal path, not just the happy path?
6. Would `contracts-ci.yml` pass on this push?

If the answer to any of these is "no" or "not sure," fix it before the push, not after.
