# Pinned toolchain versions

Both of you are on Windows. That removes a whole class of cross-OS bugs
(path separators, line-ending defaults differ less than Windows/macOS/Linux
mixed teams) — don't reintroduce them by using different tool versions.
Pin exactly these until there's a specific reason to bump, and when you do
bump, bump together in the same PR, not independently.

## Python (ML + backend track)

| Tool | Pinned version | Why this exact one |
|---|---|---|
| Python | 3.11.x | Matches what this contract scaffold was generated and tested with. Do not use 3.13 — some ML packages lag on new Python support and you'll spend hours on install errors instead of the model. |
| PyTorch | pin in `requirements.txt` once training starts, CPU or CUDA build matching your dev machine — **record the exact `pip install` line used, including `--index-url` if CUDA, in `README.md`**, because "pip install torch" resolves to a different wheel depending on OS/CUDA and is the single most common "works on my machine" bug in ML repos |
| onnx | 1.22.0 | pin exact patch version — the ONNX file format itself is stable, but `onnx.checker` behavior has changed between minors before |
| onnxruntime (Python, for training-side validation only) | 1.25.0 | **This is NOT the same package as the Android side's ONNX Runtime Mobile.** They are different builds of the same project and can support different opsets. See the opset row below — this is the row most likely to cause a "runs in Python, crashes on Android" bug if ignored. |
| fastapi | 0.141.x | |
| pydantic | 2.13.x | v2, not v1 — the `alias_generator=to_camel` API used in `contracts/backend_api/stub_api.py` is v2-only |
| uvicorn | 0.46.x | |
| **Dependency management** | Use a `requirements.txt` with `==` pins for everything (not `>=`), generated with `pip freeze > requirements.txt` inside a fresh virtualenv. Commit it. Anyone setting up the project runs `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt` and gets byte-identical package versions — not "whatever pip resolves today." |

## ONNX opset ↔ ONNX Runtime Mobile — the pairing that actually matters

`generate_stub_model.py` pins **opset 17**. Before building the Android
side against `anchor_net_stub.onnx`, confirm the ONNX Runtime Mobile /
`onnxruntime-android` AAR version you add to Gradle supports opset 17 for
every op used (`Reshape`, `Gemm` — both are old, stable ops, so this is
low-risk for the stub, but **re-check this the day the real trained model
adds anything more exotic**, e.g. a `LayerNormalization`, a custom RNN op,
or int8 quantization ops — those have had real opset/mobile-runtime support
gaps in the wild). If the real model's architecture needs a newer opset,
that is itself a `contracts/model_io` **MAJOR** version bump (see
`VERSIONING.md`) because it changes what runtime version the app must ship.

## Kotlin / Android (on-device engine + app track)

Record here once decided (fill in during setup, together):

| Tool | Pinned version | Notes |
|---|---|---|
| JDK | — | Android Studio bundles one; record the exact version `java -version` reports |
| Kotlin | — | |
| Android Gradle Plugin (AGP) | — | AGP/Gradle/Kotlin have a compatibility matrix — mismatches produce opaque Gradle sync failures. Pick versions from Android Studio's own bundled recommendation and don't hand-override unless you both agree. |
| Gradle | — | Use the Gradle wrapper (`gradlew`/`gradlew.bat`), commit `gradle/wrapper/gradle-wrapper.properties` — never rely on a locally-installed Gradle, that's exactly the "works on my machine" trap. |
| `compileSdk` / `targetSdk` / `minSdk` | minSdk 29 per PRD A4 | |
| `onnxruntime-android` (or `onnxruntime-mobile`) AAR | — | **Must be recorded and pinned in `build.gradle.kts` with an exact version, not a `+` or range.** Check its release notes for the minimum ONNX opset it supports before picking a version. |
| **Dependency management** | Use Gradle's [version catalog](https://docs.gradle.org/current/userguide/platforms.html) (`gradle/libs.versions.toml`) with exact pinned versions for every dependency, not floating `+`/`latest.release`. Commit it. |

## The rule for this whole file

Whoever changes a pinned version here does it in a PR the other person
reviews, and re-runs the relevant golden-vector test (Python side:
`pytest`; Android side: the instrumented test against
`contracts/model_io/golden_vectors/`) before merging. A version bump that
silently changes numeric behavior (a new onnxruntime patch release
changing floating-point rounding in some op, for instance) is exactly what
the golden vectors exist to catch.
