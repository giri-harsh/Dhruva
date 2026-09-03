# How contracts version, and the rule for changing one

Every contract in this folder carries its own `contract_version` /
`api_contract_version` field (semver: `MAJOR.MINOR.PATCH`), independent of
the app's version and the model's trained-weights version. Three separate
version numbers exist on purpose — a new model release does not require an
app release, and an app release does not require a new model:

| Version | Lives in | Bumped when |
|---|---|---|
| `model_io` contract version | `contracts/model_io/model_manifest.json`, embedded in the ONNX file's metadata | Window size, feature order, feature units, or output semantics change |
| `replay_csv` contract version | `contracts/replay_csv/schema.json` | A column is added, removed, renamed, reordered, or its unit/dtype changes |
| `backend_api` contract version | `contracts/backend_api/stub_api.py` (`API_CONTRACT_VERSION`), reflected in `openapi.json` | Any endpoint's request/response shape changes, an endpoint is added or removed |
| Model weights version | `GET /v1/model/version` response `modelVersion` | Every time retrained weights are published, even with an unchanged `contract_version` |

## The rule

1. **A breaking change to any file under `contracts/` requires a message to
   the other person before you write code against it, not after.** For a
   two-person team this is a Slack/WhatsApp/whatever-you-use message, not
   a formal process — but it happens *before* the PR, every time.
2. **Bump MAJOR** for anything that would silently break the other side if
   they didn't know (renamed/removed/reordered field, changed unit,
   changed shape). **Bump MINOR** for a backward-compatible addition (new
   optional field, new enum value that old code can ignore). **Bump PATCH**
   for a fix that changes no shape (e.g. correcting a description string).
3. **The receiving side declares what it supports and refuses what it
   doesn't**, rather than trying to be permissive:
   - The Android app declares `MIN_SUPPORTED_MODEL_CONTRACT_VERSION`. If
     `GET /v1/model/version` or the ONNX file's own embedded
     `contract_version` is not semver-compatible with that floor, the app
     **refuses to load the model** and falls back per FR-24 (model-failure
     fallback in the PRD), logging exactly why. It never "tries anyway."
   - The backend declares `API_CONTRACT_VERSION` in its OpenAPI spec and in
     every response's implicit version (the spec itself, fetched at
     `/openapi.json`). A client built against an incompatible major version
     should fail a build-time or startup check, not a runtime crash three
     screens into the demo.
4. **Never hand-edit a generated artifact.** `anchor_net_stub.onnx`,
   `model_manifest.json`, `golden_vectors/*.json`, `sample_replay.csv`, and
   `openapi.json` are all outputs of a script in the same folder. If a
   value in one of them needs to change, change the script's source of
   truth (`generate_stub_model.py`'s constants, `schema.json`, or
   `stub_api.py`'s Pydantic models) and re-run the generator. This is what
   makes "the doc and the code disagree" structurally impossible instead of
   just discouraged.

## Compatibility matrix (fill in as real versions ship)

| App version | Min model contract | Min API contract | Notes |
|---|---|---|---|
| 0.1.0 (initial) | 1.0.0 | 1.0.0 | stub artifacts in this repo |
