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

Filled in Week 1 (Kamal), verified against primary sources on 2026-09-03 —
**not yet build-verified**, because this dev machine has no JDK 17+, no
Android SDK, no Gradle, no `kotlinc` (only a stray JRE 8 was found). Every
version below is correct per Maven Central / Google Maven / official release
notes at the time of pinning; the FIRST `./gradlew` run on a machine with
real tooling is the actual verification and may surface something this
research couldn't. Do that run before trusting this table blindly.

| Tool | Pinned version | Notes |
|---|---|---|
| JDK | **17** | Required by AGP 9.3.0 (both min and default). Android Studio bundles a compatible JDK; if building outside Studio, install Temurin 17. **Not present on the machine this was researched on** — `java -version` there reports 1.8.0_401. Record what `java -version` reports on whichever machine actually runs the build. |
| Kotlin | **2.4.0** (latest stable; 2.4.20 was still RC3 as of 2026-09-02 — do not build on an RC) | Applied via AGP 9's **built-in Kotlin support**, not the separate `org.jetbrains.kotlin.android` plugin (that plugin is incompatible with AGP 9's new DSL). Compose Compiler is applied via `org.jetbrains.kotlin.plugin.compose` at the same 2.4.0 version, which is how JetBrains guarantees Kotlin<->Compose-Compiler pairing — no separate compiler-version lookup needed. **Open item, unverified without real tooling:** confirm what Kotlin compiler version AGP 9.3.0's built-in support actually resolves to on first sync; JetBrains' own post on AGP 9 migration describes version-specific stabilisation (e.g. "AGP 9.2.0 fully stabilises Kotlin 2.1.x compatibility"), which may mean the effective compiler version is constrained by AGP itself rather than freely selectable. |
| Android Gradle Plugin (AGP) | **9.3.0** (July 2026) | **We checked AGP 8.13.0 first, specifically because "conservative, not bleeding-edge" was the instruction — and reversed course on hard evidence, not preference:** the current stable Compose BOM (`2026.08.00`) requires **AGP >= 9.1.1**. Pinning AGP 8.x would mean either forcing an old, unverified Compose BOM to match it, or a build that fails at Compose-Compiler resolution. 9.3.0 clears that floor with margin, is one release behind the newest (9.4.0 shipped this same month, September 2026) rather than the literal bleeding edge, and -- since this is a greenfield project, not a migration -- gets AGP 9's built-in-Kotlin model for free with no legacy KGP plugin to reconcile. Max supported API level 37; min/default SDK Build Tools 36.0.0. |
| Gradle | **9.5.0**, via the wrapper only | AGP 9.3.0's own stated minimum *and* default. Commit `gradle/wrapper/gradle-wrapper.properties` -- never rely on a locally-installed Gradle. |
| `compileSdk` | **37** | AGP 9.3.0's ceiling; also what current `compose-bom:2026.08.00` targets. |
| `targetSdk` | **36** | One behind `compileSdk`, deliberately -- avoids opting into API-37-gated runtime behaviour changes (that shipped *this month*) mid-hackathon, while still compiling against the current SDK. |
| `minSdk` | **29** | Fixed by v3 PRD A4 -- not a toolchain choice. `onnxruntime-android`'s prebuilt AAR is built for API 27 minimum by default (API 24 without NNAPI), so 29 is comfortably clear of it -- no ABI/API conflict there. |
| `onnxruntime-android` AAR | **1.29.0**, exact pin | Verified against Maven Central's authoritative `maven-metadata.xml` (`<release>1.29.0</release>`, `lastUpdated 20260812`) -- **not** mvnrepository.com (stale at 1.27.0) or the legacy `search.maven.org` index (stale at 1.22.0; both are known-lagging mirrors, don't trust them for "latest"). Supports opset 17 (our contract's pin) comfortably -- ORT has supported opset 17 since 1.12/1.13, and the stub model uses only `Reshape`/`Gemm`, both old and stable. Re-verify the opset floor the day a real trunk with GroupNorm/GELU/int8-QDQ ships (TOOLCHAIN.md's Python section already flags this same row). `abiFilters` restricted to `arm64-v8a` (+ `x86_64` for the emulator only) in `:android/app` -- not all four ABIs -- to keep APK size down; irrelevant to `:core` or `:edge`, which don't bundle native libs directly. |
| ONNX Runtime -- desktop (`:core` tests, `:edge`) | **`com.microsoft.onnxruntime:onnxruntime:1.29.0`** (same version, desktop artifact, not `-android`) | `:core`'s `ModelRunner` is written against the shared `ai.onnxruntime.*` API surface (`compileOnly` at the `:core` level -- no native binaries bundled there). `:android/app` supplies `onnxruntime-android` at `implementation` scope (Android native libs); `:edge` and `:core`'s own `testImplementation` supply plain desktop `onnxruntime` (JVM native libs). This is what lets the golden-vector test run as a **plain JUnit test on a laptop**, no emulator or device needed -- the whole point of the Kotlin/JVM-core decision in v3 PRD Sec 10.2. |
| **Dependency management** | Gradle [version catalog](https://docs.gradle.org/current/userguide/platforms.html), `gradle/libs.versions.toml`, exact pins everywhere, no `+`/`latest.release`. Committed. |

## The rule for this whole file

Whoever changes a pinned version here does it in a PR the other person
reviews, and re-runs the relevant golden-vector test (Python side:
`pytest`; Android side: the instrumented test against
`contracts/model_io/golden_vectors/`) before merging. A version bump that
silently changes numeric behavior (a new onnxruntime patch release
changing floating-point rounding in some op, for instance) is exactly what
the golden vectors exist to catch.
