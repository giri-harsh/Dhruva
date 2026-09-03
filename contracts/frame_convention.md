# Frame convention & gravity handling — READ THIS BEFORE touching accel/gyro anywhere

This file exists because it didn't, and that gap caused a real question:
"is `sample_replay.csv`'s `accel_z` ≈ 9.8 (gravity-inclusive) inconsistent
with the model's `accel_z` (gravity-removed)?" Answer: no, they're two
different points in the same pipeline. This doc freezes that pipeline so
the question can't come up again.

## The short version

There are **two accelerometer representations** in this repo, and they are
**not the same signal**:

| Stage | File / contract | Frame | Gravity | Rate |
|---|---|---|---|---|
| 1. Raw sensor log | `contracts/replay_csv/` (`sample_replay.csv`) | Phone frame (or vehicle-frame *if* the producer already aligned it — see below) | **Included** (`accel_z ≈ ±9.8` at rest) | as logged, ~10 Hz smartphone stream |
| 2. Model input tensor | `contracts/model_io/` (`imu_window`, `FEATURE_ORDER`) | Vehicle frame (ISO 8855-style: x=forward, y=left, z=up) | **Removed** — this is *linear* acceleration | 10 Hz, windowed (20 samples = 2.0 s) |

Stage 1 → Stage 2 is a real preprocessing step, not a relabeling. Both
tracks must implement it **identically**, because the model was trained on
Stage-2 features and any divergence is silent numeric drift, not a crash.

## The preprocessing step, precisely

Given a raw sample with phone-frame accelerometer `accel_raw` and the
phone's own `GRAVITY` sensor channel `gravity` (both in m/s², same phone
frame):

```
linear_phone = accel_raw - gravity        # gravity removed, still phone frame
linear_vehicle = R @ linear_phone         # rotated into vehicle frame
gyro_vehicle   = R @ gyro_raw             # gyro rotated the same way (gravity doesn't apply to gyro)
```

`R` (phone-frame → vehicle-frame rotation) is derived in two steps, per the
ML track's reference implementation (`ml/anchor/data/features.py`,
`_rotation_gravity_to_down` + `_yaw_rotation` + `_best_yaw`):

1. **Roll/pitch**: rotate so the mean `gravity` vector over the sequence
   points along vehicle -z (i.e. "down" in vehicle frame is down in real
   life — corrects for how the phone is mounted/tilted, not for road
   pitch).
2. **Yaw**: rotate about the now-vertical axis by the angle that maximizes
   correlation between the rotated longitudinal accel and the vehicle's
   own longitudinal accel signal (CAN, when available) — i.e. "forward" in
   vehicle frame is aligned to actual direction of travel, not to
   wherever the phone happens to be pointed in the cabin.

This is a per-sequence (or per-mount-session) calibration, not a per-sample
computation — `R` is fit once from a window of data and then applied to
every sample in that sequence.

## Axis convention (the actual numbers)

Vehicle frame is right-handed, ISO 8855-style:

- **x** = forward (direction of travel)
- **y** = left
- **z** = up

This was already the working assumption in both split PRDs
(`PRD-ML-BACKEND.md` §6.3, `PRD-ANDROID-ENGINE.md`) and is now the
committed, implemented default — `ml/anchor/data/features.py` implements
exactly this. Treat it as **locked**, not open, as of this file's commit.

## What each contract file's fields actually mean

- **`contracts/model_io/` (`FEATURE_ORDER = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]`)**
  is **always** Stage 2: vehicle-frame, gravity-removed linear acceleration
  (m/s²) and vehicle-frame gyro (rad/s). At rest, `accel_z ≈ 0`, not `≈ 9.8`.
  If you feed raw phone-frame, gravity-inclusive samples into the model,
  you will get silently-wrong output — the graph has no way to detect this.

- **`contracts/replay_csv/schema.json`** (`sample_replay.csv`) is Stage 1:
  a **raw sensor log**, gravity-inclusive, in whatever frame the producer
  captured it in (phone frame unless a producer explicitly ran alignment
  first — the schema's `accel_x` note already says "vehicle-frame-aligned
  if alignment has run, else raw phone-frame"; in practice, for the sample
  fixture and for real phone captures, assume **not yet aligned**, i.e.
  phone frame, gravity-inclusive, until a `frame: vehicle-aligned` marker
  is added to a given CSV — no such marker exists yet, so treat every
  `replay_csv`-schema file today as Stage 1 raw).

  **This means the replay CSV is never fed directly into the model.**
  Anything that consumes `sample_replay.csv` (the on-device replay
  harness, S-06, any offline eval script) must run the Stage 1 → Stage 2
  step above — gravity removal + rotation — before windowing and calling
  the model, and must do so with the *same* alignment method the training
  pipeline used, not an approximation, or the replayed predictions will
  not match what the model produces on real live sensor input.

## Who implements what

- **ML/backend track** already has the reference implementation:
  `ml/anchor/data/features.py` (`align_sequence_to_vehicle_frame`,
  `sequence_model_features`). This is the ground truth for "what did the
  model actually see during training." Any Kotlin port must reproduce its
  arithmetic exactly (not just its intent) — same rotation math, same
  order of operations, gravity subtracted **before** rotation.
- **Android/on-device track** needs a Kotlin equivalent that: (a) reads
  `GRAVITY` (or derives it via a low-pass filter on raw accel, matching
  whichever source `features.py` actually uses — confirm this with the ML
  track before implementing, since a low-pass approximation and the
  phone's own gravity sensor can disagree slightly), (b) computes the same
  `R`, (c) applies it identically to accel and gyro, before ever calling
  the on-device ONNX model or comparing against a golden vector.

## Golden vectors don't cover this yet — known gap

`contracts/model_io/golden_vectors/*.json` currently contain synthetic
inputs (`stationary`, `accelerating_straight`, `random_vibration`) that are
already Stage-2 (vehicle-frame, gravity-removed) by construction — they
don't exercise the Stage 1 → Stage 2 conversion at all, because the stub
model doesn't do that conversion (the model only ever sees Stage 2 input).
**The Stage 1 → Stage 2 conversion itself has no cross-track golden test
yet.** Recommended before Week-2 integration: add a fixture that takes one
short raw `sample_replay.csv`-shaped sequence, runs it through
`features.py`'s alignment on the Python side, and commits the resulting
Stage-2 window as a new golden vector category (`replay_to_model_input`)
so the Android port can be checked against it byte-for-byte, not just "it
compiles."

## Contract version note

This file documents behavior that was already implemented in
`ml/anchor/data/features.py` before this file existed — it is not a
contract *change*, so `model_io` `CONTRACT_VERSION` stays `1.0.0`. It
becomes a real version-relevant change only if the alignment algorithm
itself (gravity source, yaw-search method, rotation order) changes later,
per `../VERSIONING.md`'s MAJOR-bump rule.
