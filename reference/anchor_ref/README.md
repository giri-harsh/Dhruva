# `reference/anchor_ref/` — Python reference for the on-device engine

Harshit maintains this; **Kamal's Kotlin `core/` must reproduce its arithmetic**
(not just its intent). Two consumers, per PRD §5.3 / §10.5:

1. **Baseline / ablation harness** (`ml/anchor/bench/`) runs B2 and B3 through
   here so every baseline uses identical inputs, ground truth, and metrics (§6.3).
2. **Golden-vector generation** for Kamal's Kotlin regression tests
   (`reference/golden/`).

## What exists

| | file | status |
|---|---|---|
| **B2** strapdown INS, no learning | `strapdown.py` | ✅ complete, runnable |
| **B3** ESKF + NHC + ZUPT | `eskf.py` | ✅ complete, runnable (Kamal, `android/week3-reference-eskf`) |

`import reference.anchor_ref as ref` exposes `ref.strapdown_dead_reckon` always
and `ref.eskf_dead_reckon` + `ref.HAS_ESKF == True` now that `eskf.py` has
landed. The harness's `B3Eskf.runnable` flips automatically — see
`ml/tests/test_eskf.py` for scenario coverage (stationary/ZUPT,
straight/turning NHC, gated velocity fusion, chi-square rejection) and
`ml/tests/test_reference_golden.py::test_eskf_golden_vectors_reproduce`
for the regression guard.

**`eskf.py`'s state is not a full 3-D port of `core/.../fusion`'s 15-state
design** — `feat_window` is gravity-REMOVED linear acceleration (identical
to what ANCHOR-Net consumes), so there is no gravity signal to mechanize or
correct 3-D attitude from. The state is `strapdown.py`'s own reduced
(east, north, v_east, v_north, heading), extended with exactly the states a
genuine ESKF needs: accelerometer bias (forward, lateral) and gyroscope
bias (yaw). Full derivation, the fixed heading-rate sign this function uses
in place of `strapdown_dead_reckon`'s own per-call calibration (this
interface carries no pre-outage data to calibrate against), and a real bug
found in `strapdown.py`'s own body→world rotation while building this (not
fixed here — flagged for a separate decision) are documented in
`eskf.py`'s own module docstring.

## The interface `eskf.py` must provide (B3)

```python
def eskf_dead_reckon(
    feat_window: np.ndarray,     # [T, 6] aligned vehicle-frame model input:
                                 #   accel_x(fwd), accel_y(left), accel_z(up)  [linear m/s^2]
                                 #   gyro_x(roll), gyro_y(pitch), gyro_z(yaw)  [rad/s]
                                 #   — IDENTICAL to what ANCHOR-Net receives (contracts/model_io,
                                 #     contracts/frame_convention.md)
    *,
    dt_s: float,                 # 0.1 (10 Hz)
    v0_mps: float,               # last GNSS speed before the outage
    heading0_rad: float,         # last GNSS heading, compass (CW from north)
    # optional, for the "+ velocity head" ablation rows (5, 6, 8):
    vel_mean_mps: np.ndarray | None = None,   # ANCHOR-Net Head A, per inference window
    vel_logvar: np.ndarray | None = None,     # ANCHOR-Net Head B (log variance)
) -> dict:
    """-> {"east_m": np.ndarray[T+1], "north_m": np.ndarray[T+1],
           "heading_end_rad": float}  in a local ENU frame, origin at the
    outage start. No ground-truth / GNSS reads during the outage."""
```

The ablation runner will call it three ways to fill PRD §6.7's rows:
- **row 3 (B3)**: `vel_mean_mps=None` — filter runs on NHC/ZUPT alone.
- **row 4**: `vel_mean_mps=<head A>`, treated with a fixed measurement noise `R`.
- **row 5 (primary claim)**: `vel_mean_mps` + `vel_logvar` → per-window
  `R = exp(vel_logvar)`.

Keep `feat_window` the sole dynamic input and everything else a scalar/array
argument, so a Kotlin port has a clean signature to match.

## B2 contract (already implemented — see `strapdown.py` docstring)

State `(east, north, v_east, v_north, heading)`; per 10 Hz step, in this order:
`heading += sign·gyro_z·dt` ; `a_world = Rz(heading)·[accel_x, accel_y]` ;
`v += a_world·dt` ; `p += v·dt`. `sign` is resolved once from the pre-outage
GNSS heading rate vs `gyro_z` correlation (last-known calibration; no CAN).
