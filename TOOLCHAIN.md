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

Filled in Week 1 (Kamal). **Updated 2026-09-03, second pass:** a real JDK
17 + Android SDK were installed on this machine (see "Real environment
setup," below) and `:core` — the whole engine, 33 main classes / 13 test
classes — now genuinely compiles and its full test suite genuinely runs
and passes, including the ONNX golden-vector test against the real
committed model. **Gradle itself cannot execute in this specific
environment** (root-caused, see "Known environment blocker: Gradle daemon
startup," below) — every number in this table is now verified either by
real `kotlinc`/`java` execution (`:core`, `:edge`) or remains
research-only, pinned but unbuilt (`:android/app`, which needs AGP's full
pipeline — aapt2, manifest merging, D8 — with no honest manual substitute).
The FIRST real `./gradlew` run on a machine without this specific blocker
is still the actual verification of the Gradle *build graph itself*
(module wiring, plugin resolution) — this table's version numbers are
now double-verified (research + manual execution), but the Gradle
configuration that reads them has not been.

| Tool | Pinned version | Notes |
|---|---|---|
| JDK | **17** | Required by AGP 9.3.0 (both min and default). **Installed this session**: Microsoft Build of OpenJDK 17.0.20.1+1, portable zip from `aka.ms/download-jdk` (not the winget/MSI path — that requires interactive UAC elevation this environment can't provide; the winget attempt was left running and completed on its own much later, so a second, redundant system install may now also exist — harmless, but worth `Get-Package Microsoft.OpenJDK.17` and removing the duplicate before this becomes confusing). Verified: `java -version` → `openjdk version "17.0.20.1" 2026-08-18 LTS`. |
| Kotlin | **2.4.0** (latest stable; 2.4.20 was still RC3 as of 2026-09-02 — do not build on an RC) | Applied via AGP 9's **built-in Kotlin support**, not the separate `org.jetbrains.kotlin.android` plugin (that plugin is incompatible with AGP 9's new DSL). Compose Compiler is applied via `org.jetbrains.kotlin.plugin.compose` at the same 2.4.0 version, which is how JetBrains guarantees Kotlin<->Compose-Compiler pairing — no separate compiler-version lookup needed. **Open item, unverified without real tooling:** confirm what Kotlin compiler version AGP 9.3.0's built-in support actually resolves to on first sync; JetBrains' own post on AGP 9 migration describes version-specific stabilisation (e.g. "AGP 9.2.0 fully stabilises Kotlin 2.1.x compatibility"), which may mean the effective compiler version is constrained by AGP itself rather than freely selectable. |
| Android Gradle Plugin (AGP) | **9.3.0** (July 2026) | **We checked AGP 8.13.0 first, specifically because "conservative, not bleeding-edge" was the instruction — and reversed course on hard evidence, not preference:** the current stable Compose BOM (`2026.08.00`) requires **AGP >= 9.1.1**. Pinning AGP 8.x would mean either forcing an old, unverified Compose BOM to match it, or a build that fails at Compose-Compiler resolution. 9.3.0 clears that floor with margin, is one release behind the newest (9.4.0 shipped this same month, September 2026) rather than the literal bleeding edge, and -- since this is a greenfield project, not a migration -- gets AGP 9's built-in-Kotlin model for free with no legacy KGP plugin to reconcile. Max supported API level 37; min/default SDK Build Tools 36.0.0. |
| Gradle | **9.5.0**, via the wrapper only | AGP 9.3.0's own stated minimum *and* default. Commit `gradle/wrapper/gradle-wrapper.properties` -- never rely on a locally-installed Gradle. |
| `compileSdk` | **36** — **changed from 37, documented here per this session's own rule** | Old: `37` (research-only pin, based on "AGP 9.3.0's ceiling is API 37"). New: `36`. Reason: real `sdkmanager --list` against the live repository (not research) showed Android has moved to a **sub-versioned API 37** (`platforms;android-37.0`, `37.1`, `37.2` — no plain `platforms;android-37` exists any more), which postdates whatever AGP-9.3.0-era documentation my original research read. Rather than guess which `37.x` a bare `compileSdk = 37` Kotlin-DSL integer resolves to under AGP 9.3, installed the clean, unambiguous `platforms;android-36` instead and moved `compileSdk` down to match. This also means `compileSdk` now equals `targetSdk` (see next row) — both conservative, both real, both installed and present in the local SDK. If a later real Gradle sync needs 37 specifically (e.g. a Compose API only in 37.x), install the correct `37.x` sub-version explicitly and bump both rows together, in one PR, per this file's own rule below. |
| `targetSdk` | **36** | One behind the *original* `compileSdk=37` research pin; now equal to the corrected `compileSdk=36` above. Still the right choice even so: avoids opting into brand-new API-level-gated runtime behaviour changes mid-hackathon. |
| `minSdk` | **29** | Fixed by v3 PRD A4 -- not a toolchain choice. `onnxruntime-android`'s prebuilt AAR is built for API 27 minimum by default (API 24 without NNAPI), so 29 is comfortably clear of it -- no ABI/API conflict there. |
| `onnxruntime-android` AAR | **1.29.0**, exact pin | Verified against Maven Central's authoritative `maven-metadata.xml` (`<release>1.29.0</release>`, `lastUpdated 20260812`) -- **not** mvnrepository.com (stale at 1.27.0) or the legacy `search.maven.org` index (stale at 1.22.0; both are known-lagging mirrors, don't trust them for "latest"). Supports opset 17 (our contract's pin) comfortably -- ORT has supported opset 17 since 1.12/1.13, and the stub model uses only `Reshape`/`Gemm`, both old and stable. Re-verify the opset floor the day a real trunk with GroupNorm/GELU/int8-QDQ ships (TOOLCHAIN.md's Python section already flags this same row). `abiFilters` restricted to `arm64-v8a` (+ `x86_64` for the emulator only) in `:android/app` -- not all four ABIs -- to keep APK size down; irrelevant to `:core` or `:edge`, which don't bundle native libs directly. |
| ONNX Runtime -- desktop (`:core` tests, `:edge`) | **`com.microsoft.onnxruntime:onnxruntime:1.29.0`** (same version, desktop artifact, not `-android`) | `:core`'s `ModelRunner` is written against the shared `ai.onnxruntime.*` API surface (`compileOnly` at the `:core` level -- no native binaries bundled there). `:android/app` supplies `onnxruntime-android` at `implementation` scope (Android native libs); `:edge` and `:core`'s own `testImplementation` supply plain desktop `onnxruntime` (JVM native libs). This is what lets the golden-vector test run as a **plain JUnit test on a laptop**, no emulator or device needed -- the whole point of the Kotlin/JVM-core decision in v3 PRD Sec 10.2. |
| **Dependency management** | Gradle [version catalog](https://docs.gradle.org/current/userguide/platforms.html), `gradle/libs.versions.toml`, exact pins everywhere, no `+`/`latest.release`. Committed. |
| ArchUnit | **1.5.0** — **changed from 1.3.0, documented here** | Old: `1.3.0` (a guess, never checked against Maven Central — every *other* version in this table was checked, this one slipped through). New: `1.5.0`, confirmed via `archunit-junit5`'s own `maven-metadata.xml` (`<release>1.5.0</release>`). The wrong guess surfaced as a real runtime failure — see below. Also: `TierDependencyTest` calls ArchUnit's plain Java API directly (`ClassFileImporter`, `noClasses()`) from an ordinary `@Test` method, not ArchUnit's own `@AnalyzeClasses`-based JUnit 5 engine — so it only needs the core `archunit` artifact, not `archunit-junit5`/`archunit-junit5-api`/`archunit-junit5-engine` at all. `libs.versions.toml` and `core/build.gradle.kts` updated accordingly (dependency simplified, not just re-pinned). |

## Real environment setup, this session (2026-09-03)

The previous pass correctly reported no JDK 17+/Android SDK/Gradle on this
machine. That gap is now closed, using only official tooling, none of it
requiring admin elevation (the winget/MSI path hit an interactive UAC
prompt with nobody able to click it — abandoned in favour of portable
zips, which is the standard pattern for a project-local toolchain anyway):

- **JDK 17**: Microsoft Build of OpenJDK 17.0.20.1+1, portable zip from
  `aka.ms/download-jdk`, extracted to `D:\jdk17\`.
- **Android SDK**: command-line tools `commandlinetools-win-15859902`
  from `dl.google.com` (sha256 verified against the officially published
  checksum before extracting), installed to `D:\android-sdk\`.
  `platform-tools`, `platforms;android-36`, `build-tools;36.0.0`
  installed via `sdkmanager`; all licenses accepted non-interactively.
  `local.properties` (gitignored) points `sdk.dir` at this.
- **Gradle 9.5.0**: official binary distribution from
  `services.gradle.org`, sha256-verified against the checksum already
  committed in `gradle/wrapper/gradle-wrapper.properties`, extracted to
  `D:\gradle-dist\`. This is what the wrapper generation attempted to use
  — see the blocker below for why that specific step still failed.

## Known environment blocker: Gradle daemon startup fails on this machine

**Every Gradle invocation in this environment — `wrapper`, `clean`,
`test`, `assembleDebug`, even `--version` — fails identically**, before
reaching any project-specific code, with:

```
java.io.IOException: Unable to establish loopback connection
Caused by: java.net.SocketException: Invalid argument: connect
  at sun.nio.ch.UnixDomainSockets.connect0(Native Method)
```

**Root cause, confirmed by full stack-trace analysis, not guessed:**
Gradle's client-daemon IPC always calls `java.nio.channels.Selector.open()`
(true even for a "single-use" daemon under `--no-daemon`). On this
specific JDK/Windows combination, `Selector.open()` internally calls
`Pipe.open()` for its self-wakeup mechanism, and this JDK's `Pipe.open()`
implementation uses an **AF_UNIX loopback socket** internally — which
fails here with `EINVAL`, even though plain TCP loopback (`ServerSocket`/
`Socket`, confirmed separately with a standalone test) works perfectly
fine, and the `afunix.sys` driver is confirmed present and running at the
OS level.

**This was diagnosed exhaustively before being accepted as a real
blocker, not a config problem to keep chasing:**
- Reproduces identically across 2 JDK vendors (Microsoft, Eclipse
  Temurin) and 2 patch levels (17.0.13 from Oct 2024, 17.0.20.1 from Aug
  2026) — ruling out "bad JDK build."
- Reproduces identically via both Git Bash and native PowerShell, with
  the harness's own sandbox explicitly disabled on both — ruling out a
  tool-specific wrapper issue.
- Reproduces identically when forcing the classic
  `sun.nio.ch.WindowsSelectorProvider` instead of the newer
  WEPoll-based one via `-Djava.nio.channels.spi.SelectorProvider=...` —
  ruling out "just pick a different selector," since the failure is one
  level deeper, in `PipeImpl` itself, shared by both providers.
- Matches a known, externally-tracked issue
  ([anthropics/claude-code#41432](https://github.com/anthropics/claude-code/issues/41432))
  describing this exact `WEPollSelectorImpl` loopback failure for Java
  child processes spawned by Claude Code / Claude Desktop on Windows —
  this is the most likely actual cause: a limitation of *this specific
  execution environment*, not of the project, the JDK choice, or the
  Gradle configuration.

**What this means practically: nobody should spend more time chasing
this specific symptom in a Claude Code Windows session.** It is very
likely to *not* reproduce on a normal Windows dev machine, in Android
Studio, or in CI — those aren't spawned the same way. Try a real
`./gradlew` there first. If it also fails there with this exact stack
trace, `-Djava.nio.channels.spi.SelectorProvider=sun.nio.ch.WindowsSelectorProvider`
did NOT fix it here but is worth trying in a genuinely different
environment, since the deeper `PipeImpl` AF_UNIX issue may not be present
outside whatever is specific to this one.

## How `:core` and `:edge` were verified without a working Gradle

Since the Gradle *build tool* is blocked but the underlying JDK/Kotlin
*compiler and runtime* are not (`kotlinc` and `java` both run cleanly
here, confirmed with a trivial smoke-test jar first), `:core` and `:edge`
were compiled and tested by hand, replicating what Gradle's Kotlin plugin
does internally:

1. Every dependency jar in `gradle/libs.versions.toml` fetched directly
   from Maven Central at the exact pinned version (not `latest` — the
   exact coordinate the catalog names).
2. `:core/src/main` compiled with `kotlinc` against those jars.
3. `:core/src/test` compiled against `:core`'s compiled output plus
   `-Xfriend-paths=<main output>` — the same flag Gradle's own Kotlin
   plugin passes internally to make `internal` declarations in `main`
   visible to `test`, so this reproduces real Gradle visibility
   semantics, not a looser manual approximation.
4. Tests executed via the official `junit-platform-console-standalone`
   launcher (no Gradle needed to run JUnit 5 — this is a real, supported,
   standalone JUnit distribution channel).
5. `:edge` compiled and run for real against the actual committed
   `contracts/replay_csv/sample_replay.csv`.

**What this does NOT verify:** `:android/app`. There is no honest manual
substitute for AGP's resource processing (`aapt2`), manifest merging, R
class generation, or D8 dexing — those are real, non-trivial build steps
this approach cannot replicate by hand, and claiming otherwise would be
exactly the "fabricated green result" this whole exercise was meant to
avoid. `:android/app`'s Kotlin source is written and believed correct by
the same standard as everything else before this pass (matches the
`:core` APIs it calls, which ARE now verified) — but its own compilation
is genuinely unverified. It also does not verify the **Gradle build
graph itself** (module dependency wiring, plugin application, version
catalog resolution) — only the *code* the version catalog points at.

## The rule for this whole file

Whoever changes a pinned version here does it in a PR the other person
reviews, and re-runs the relevant golden-vector test (Python side:
`pytest`; Android side: the instrumented test against
`contracts/model_io/golden_vectors/`) before merging. A version bump that
silently changes numeric behavior (a new onnxruntime patch release
changing floating-point rounding in some op, for instance) is exactly what
the golden vectors exist to catch.
