# Units contract — non-negotiable, everywhere

This project has exactly one internal unit system. Any value entering or
leaving a contract boundary (CSV, ONNX model I/O, backend API JSON) uses
these units. If a third-party library or an Android sensor API hands you
something else, convert at the boundary and never carry the raw unit
further into the codebase.

| Quantity | Unit | Notes |
|---|---|---|
| Linear acceleration | m/s² | Android `SensorEvent.values` for `TYPE_ACCELEROMETER` is already m/s² — no conversion needed there |
| Angular rate (gyro) | rad/s | Android `TYPE_GYROSCOPE` is already rad/s. **IO-VNBD's smartphone gyroscope columns are documented as yaw/pitch/roll in the paper's own units — verify per-column before use; do not assume rad/s from the column name alone.** |
| Magnetic field | µT (microtesla) | Android `TYPE_MAGNETIC_FIELD` is already µT |
| Speed / velocity | m/s | Convert from km/h (÷3.6) or mph at the boundary, immediately, never downstream |
| Position | decimal degrees (lat/lon), WGS84 | |
| Distance | metres | |
| Time | milliseconds since Unix epoch, UTC, `int64` | Never a float; never local time; never seconds unless a field name says `_s` explicitly |
| Heading / bearing / yaw | radians internally in the filter state; **degrees at any human-facing or map-facing boundary** (OSM bearings, UI display) — the boundary conversion point must be a single named function, not ad hoc `* 180/pi` scattered through the code |
| Angles generally | radians in all math, degrees only at the UI/OSM boundary | Same rule as heading |

## The one bug that will happen if this file is ignored

Gyroscope in deg/s instead of rad/s is a ~57x scale error. It does not
crash anything. The filter will run, produce a trajectory, and be
silently, catastrophically wrong — exactly the kind of bug that "looks
like a bad model" and wastes a day of debugging the wrong layer.

**Defensive rule, enforced in code on both sides, not just documented
here:** any function that ingests raw gyro data asserts `abs(value) < 10`
rad/s (≈573°/s — no road vehicle yaws that fast) before using it. See
`contracts/replay_csv/validate_replay_csv.py` for the reference
implementation of this check; the Kotlin sensor-ingestion code must carry
the equivalent assertion.

## Where this contract is enforced automatically

- `contracts/replay_csv/schema.json` and its validator — CSV boundary
- `contracts/model_io/model_manifest.json` — model I/O boundary (`feature_units`)
- Backend API — all API responses use SI units per this table; Pydantic
  field names do not repeat the unit (e.g. `speed`, not `speedMps`) *except*
  where ambiguity is possible across the wire, per the actual stub in
  `contracts/backend_api/stub_api.py` — when in doubt, name the field with
  its unit suffix (`sizeBytes`, `windowDurationS`) rather than relying on
  a reader to already know.
