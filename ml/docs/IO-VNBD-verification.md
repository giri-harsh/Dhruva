# IO-VNBD — Day 1 empirical verification

**Date:** 2026-09-03
**Verifier:** Harshit (ML/backend track)
**Method:** shallow `git clone` with `GIT_LFS_SKIP_SMUDGE=1` (LFS pointers carry exact
byte sizes), then targeted `git lfs pull` of one synchronised pair
(`S (Driver A)/S3b`) to read real headers and rows.
**Repo:** `github.com/onyekpeu/IO-VNBD` @ `master`, LFS-backed, ~2.23 GB total resolved.
Local checkout (pointers only, 3.2 MB): `data/raw/IO-VNBD/` (git-ignored).

This file resolves the PRD's `[VERIFY]` items for the dataset. Where the PRD guessed,
the measured value wins and is recorded here.

---

## 1. Does the synchronised subset exist and how big is it? (R-01 — gates the whole supervised plan)

**Yes. It is large enough. The supervised plan is viable; the two-stage pretrain
fallback (§6.6) is a safety net, not the primary path.**

The repo has two top-level trees, exactly as the PRD's §6.1 says:

| Tree | Contents | Resolved size | CSV count |
|---|---|---|---|
| `Synchronised V abd S datasets/` (sic — "abd") | phone+CAN recorded at the same time | **429 MB unique** (+ a 204 MB zip duplicate, + JPodometer thumbnails) | 144 unique (72 `V-*` + 72 `S-*`) |
| `Unsynchronised V and S Dataset/` | phone-only / CAN-only, not time-locked | ~950 MB | ~430 |

Inside the synchronised tree the **same data is shipped twice**:
- `Categorised IOVNB Dataset/` — foldered by driver (`M`, `S`, `Vf`, `Vta`, `Vtb`, `Vw`, `Y`)
- `Uncategorised IOVNB Dataset/` — flat `S-Dataset/` + `V-Dataset/`

We use **Categorised** as the source of truth (driver identity is in the path).

### Synchronised sequence inventory (Categorised)

| Driver folder | Driver | # sequences (V+S pairs) | Notable |
|---|---|---|---|
| `M (Driver B)` | B | 1 (`M`) | 20 MB phone / 22 MB CAN — one long drive |
| `S (Driver A)` | A | 6 (`S1, S2, S3a, S3b, S3c, S4`) | ~124 MB total |
| `Y (Driver D)` | D | 1 (`Y1`) | 13 MB / 15 MB |
| `Vf (Driver E)` | E | 2 (`Vfa01, Vfa02`) | |
| `Vta (Driver E)` | E | ~28 (`Vta1a, Vta1b, Vta2…Vta30`, gaps at 18) | mostly short (10 kB–5 MB) |
| `Vtb (Driver E)` | E | 12 (`Vtb1…Vtb12`) | |
| `Vw (Driver E)` | E | ~20 (`Vw1…Vw17`, some a/b/c splits) | |
| **Total** | **A,B,D,E** | **~70 synchronised pairs** | |

### The catch — this is effectively ONE vehicle, and driver coverage is lop-sided (R-02)

- IO-VNBD is a **single instrumented research vehicle**. The "drivers" are different
  people in the **same car**. **No vehicle-diversity claim can be made from IO-VNBD.**
  Domain-shift to other vehicles (R-04) stays an untested risk we state openly.
- Within the *synchronised* subset the driver split is extremely unbalanced:
  **Driver E ≈ 62 of ~70 sequences.** Drivers A (6), B (1), D (1) are thin.

**What the leakage-safe split can honestly claim (see `ml/splits/`):**
- A robust **unseen-route, same-driver (E)** held-out test — plenty of E routes.
- A **held-out-driver** test is possible but small: Driver A (6 seqs) is the only
  non-E driver with enough data to be a validation/OOD-driver signal; B and D
  (1 seq each) are spot-checks, not distributions.
- Headline in-distribution numbers will be **unseen-route** on Driver E. The
  unseen-*driver* number (Driver A held out entirely) is reported separately and
  labelled as the thinner signal it is.
- France/Nigeria OOD (§6.2 row 4) lives in the **unsynchronised** phone tree only —
  no wheel-speed labels there, so OOD is GNSS-label evaluation, reported separately.

---

## 2. Sampling rates (PRD §I.1 `[VERIFY from Tables 3–4]`)

| Stream | Column giving time | Observed spacing | Rate |
|---|---|---|---|
| Smartphone (`S-*`) | `TIME SINCE START (ms)` | 2503320 → 2503421 → 2503520 (Δ ≈ 99–101 ms, jittery) | **10 Hz** ✓ |
| Smartphone GPS | `GPS LATITUDE` etc. | identical lat/lon repeated ~10 rows then steps | **~1 Hz, forward-filled** onto the 10 Hz grid ✓ |
| Vehicle (`V-*`) | `Time Since Start of Day (seconds)` | 68582.8 → 68582.9 → 68583.0 (Δ = 0.1 s exact) | **10 Hz** ✓ |
| Vehicle GPS (VBOX) | `Latitude`/`Longitude` | changes every row | **10 Hz** (VBOX ground truth) |

No stream above 10 Hz exists — high-rate pre-filtering claims must be validated on our
own captures, not IO-VNBD (say so wherever reported, per §6.4).

---

## 3. Column schemas (measured from `S (Driver A)/S3b`)

### Smartphone `S-*.csv` — 24 columns, header has irregular spacing + trailing spaces

```
GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), GPS SPEED (Kmh),
GPS ACCURACY (m), GPS ORIENTATION (°), GPS SATELLITES IN RANGE,
TIME SINCE START (ms), DATE (YYYY-MO-DD HH-MI-SS_SSS),
ACCELEROMETER X (m/s²), ACCELEROMETER Y (m/s²), ACCELEROMETER Z (m/s²),
GRAVITY X (m/s²), GRAVITY Y (m/s²), GRAVITY Z (m/s²),
GYROSCOPE Yaw (rad/s), GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s),
MAGNETIC FIELD X (µT), MAGNETIC FIELD Y (µT), MAGNETIC FIELD Z (µT),
ORIENTATION (Yaw) (°), ORIENTATION (Pitch) (°), ORIENTATION (Roll ) (°)
```

- `ACCELEROMETER *` **includes gravity** (Z ≈ 9.6–10.1 at rest-ish). A separate
  `GRAVITY *` channel is provided → linear accel = ACCEL − GRAVITY if needed.
- `GYROSCOPE Yaw/Pitch/Roll` are **rad/s** — confirmed, values ~0.01–0.03.
  `contracts/units.md`'s "verify per-column, don't assume rad/s" — **phone gyro is
  genuinely rad/s.** (The CAN yaw rate is NOT — see below.)
- `SATELLITES IN RANGE` is a string like `16 / 18` — parse the first int.
- `DATE` is **local time** (BST here); the vehicle clock is **UTC**. In S3b the phone
  reads `20:03:03` local = `19:03:03` UTC; vehicle `Time Since Start of Day = 68582.8 s`
  = `19:03:02.8` UTC. Offset ≈ 0.2 s. Good enough that row-index alignment holds
  (see §4), but the joiner should still verify, not assume.

### Vehicle `V-*.csv` — 29 columns

```
No of GPS Satellites Available, Time Since Start of Day (seconds),
Latitude (degrees), Longitude (degrees), Velocity (km/hr), Heading (degrees),
Height (km), Vertical velocity (km/hr), Sample period (seconds),
Steering Angle (degrees),
Wheel Speed Front Left (rad/sec), Wheel Speed Front Right (rad/sec),
Wheel Speed Rear Left (rad/sec), Wheel Speed Rear Right (rad/sec),
Yaw Rate (deg/sec), Indicated Vehicle Speed (km/hr),
Indicated Longitudinal Acceleration (g), Indicated Lateral Acceleration (g),
Handbrake (0/1), Gear Requested (1-5), Gear (1-5), Engine Speed (rev/min),
Coolant Temperature (deg), Clutch Position (0/1), Brake Pressure (psi),
Brake Position (0/1), Battery Voltage (volts), Air Temperature (deg),
Accelerator Pedal Position (0/1)
```

- **All four wheel-speed channels present, in rad/s** (§6.4 label recipe is viable).
  Values ~41 rad/s at ~41 km/h.
- **`Yaw Rate` is `deg/sec`** — the exact `units.md` ~57× trap, on the CAN side.
  Convert `× π/180` at load. (Contrast: phone gyro is rad/s.)
- **`Indicated Longitudinal/Lateral Acceleration` are in `g`** — convert `× 9.80665`.
- `Height` is **km**, `Velocity`/`Vertical velocity`/`Indicated Vehicle Speed` are **km/hr**.
- `Latitude`/`Longitude` here = **VBOX GPS = ground truth** (10 Hz, metre-class).
- `Sample period (seconds)` = 0.1 on every row.

---

## 4. Are `S-*` and `V-*` row-aligned in the synchronised set?

**Yes, for S3b: both files have exactly 6814 data rows.** The synchronised set appears
to be **row-index aligned** (both resampled to a common 10 Hz grid, near-identical
start/stop). The `ml/data/` "synchronised joiner" is therefore mostly:
1. load both, strip/normalise header names, convert units at the boundary,
2. assert `len(S) == len(V)` (or align on the ≤1-sample offset if not),
3. assert phone `TIME SINCE START` delta grid and vehicle `Time Since Start of Day`
   delta grid agree in length and cadence,
4. concat on row index.

**To confirm generality:** re-check `len(S)==len(V)` on every pair when the loader is
built; log any pair where it fails and handle by cross-correlation alignment on
speed (phone `GPS SPEED` vs vehicle `Indicated Vehicle Speed`).

---

## 5. Encoding / parsing gotchas (feed these into `ml/data/`)

| Gotcha | Handling |
|---|---|
| Files are **cp1252 / latin-1**, not UTF-8 (`m/s²`, `µT`, `°` mojibake as `m/s�`) | `pd.read_csv(..., encoding="cp1252")` |
| Header has `, ` and `,` mixed + trailing spaces (`ORIENTATION (Roll )`) | `skipinitialspace=True`, then `df.columns = [c.strip() for c in df.columns]`, map to canonical short names |
| `SATELLITES IN RANGE` = `"16 / 18"` | custom parse |
| Two different clocks (phone local ms-since-start, vehicle UTC s-since-midnight) | never mix; build our own `timestamp_ms` grid from row index × 100 ms anchored at 0 |
| `Vta` folder names vs file names disagree in case (`Vta02/` contains `V-vta2.csv`) | glob case-insensitively, derive sequence id from folder not filename |

---

## 6. Wheel radius (PRD §6.4 `[VERIFY: derive by regression, don't look it up]`)

Quick sanity from S3b row 0: wheel speed ≈ 41.26 rad/s, `Indicated Vehicle Speed`
= 41.31 km/h = 11.475 m/s → r ≈ 11.475 / 41.26 ≈ **0.278 m** (plausible car tyre).
**Action:** regress `mean(4 wheel ω)` against VBOX `Velocity` (converted to m/s) over
straight, GNSS-clean stretches across all synchronised sequences; take the slope as
`r`, and per-sequence residual spread feeds `label_sigma_m`. One vehicle ⇒ one radius,
but fit per-sequence too and check drift (tyre pressure/temperature).

---

## 7. Decisions this locks in

1. **Supervised path is primary.** ~70 synchronised pairs, hundreds of MB, real
   4-wheel rad/s labels. Two-stage GNSS-pretrain is the R-01 fallback only.
2. **Split axis = route, not driver, for the headline number** (Driver E dominance).
   Held-out-driver (A) reported as a separate, thinner signal. Document both in
   `ml/splits/README.md`.
3. **Unit conversions belong in one place** — `ml/data/units.py`, applied the moment
   a raw IO-VNBD frame is loaded, with the `abs(gyro) < 10 rad/s` assertion from
   `contracts/units.md` on every gyro/yaw-rate channel *after* conversion.
4. **Frame convention (open decision #1 with Kamal):** phone accel/gyro are
   device-frame; `GRAVITY` + `ORIENTATION` columns let us estimate roll/pitch, GPS
   course vs vehicle `Heading` gives yaw. Proposed ISO-8855 vehicle frame
   (x-fwd, y-left, z-up). To be written into `contracts/frame_convention.md`.
5. **`velocity_mean_mps` vs displacement (open decision #2):** §6.4 wants displacement
   in metres over the window; the scaffold field is `velocity_mean_mps`. Leaning
   toward **carry mean speed (m/s) on the wire** (= displacement / window_duration_s)
   so the field name stays honest and no `contract_version` MAJOR bump is needed —
   the filter multiplies by dt itself. Final call pending Kamal sync; recorded in
   `contracts/VERSIONING.md` compatibility notes when settled.

---

## 8. Phone-mount quality — the second data risk, measured (added 2026-09-03)

After building the loader and scoring all 72 synchronised sequences
(`ml/anchor/data/quality.py`, committed as `sequence_index.json`), a second
limitation surfaced that R-02 only half-anticipated:

**The phone in IO-VNBD's synchronised drives is mounted inconsistently, and in
many sequences its IMU carries almost no vehicle-motion information.**

Time alignment is fine — phone-GPS-speed vs vehicle-speed correlates 0.7–1.0 at
zero lag for 65 of 72 sequences. The problem is signal, not sync:

| Signal (zero-lag, 10 Hz) | S1 (Driver A) | typical Driver E | Y1 (Driver D) |
|---|---|---|---|
| `corr(rolling-std\|phone accel\|, veh speed)` — the thesis signal | 0.53 | 0.0–0.4 | 0.03 |
| `R²(veh yaw rate ~ 3 phone gyro axes)` — mount rigidity | 0.89 | ~0.1 | 0.00 |

Usability tiers (`quality.py`): **use** = vib↔speed ≥ 0.35 or yaw-R² ≥ 0.5;
**weak** = ≥ 0.20 / 0.30; **drop** otherwise or vehicle basically parked.

| Tier | Sequences | Hours | By driver |
|---|---|---|---|
| **use** | 32 | **13.9 h** | A 3.2, B 2.9, E 7.9 |
| weak | 11 | 5.4 h | A 2.6, E 2.8 |
| drop | 29 | 10.4 h | A 2.8, **D 2.0 (all of Driver D)**, E 5.6 |

### Consequences — decisions this drives

1. **The clean supervised pool is ~14 h (use) / ~19 h (use+weak), ~80 % Driver E.**
   Not "too small to train" but not driver-diverse and not large.
2. **Driver D is unusable** (Y1's phone stream looks decoupled from the vehicle).
   Held-out-driver evaluation now rests on **Driver A only** (~3 h clean / ~6 h
   total). Driver B is a single long drive. This is a real limit on what a
   generalisation claim can say — stated in `ml/splits/README.md`.
3. **The two-stage plan (§6.6) is promoted from fallback to primary architecture:**
   pre-train on the ~58 h *unsynchronised* smartphone data with GNSS-derived
   speed as a weak 1 Hz label, then fine-tune on the ~14 h clean synchronised
   data with wheel-speed labels. This is not the R-01 "sync subset too small"
   trigger firing — it's R-02 (mount inconsistency) making the weakly-supervised
   pre-train necessary for the model to see enough varied phone behaviour.
4. **`usability` and the per-sequence quality scores feed `label_sigma_m`** — a
   window from a `weak` sequence trains the variance head against a wider label
   uncertainty, which is correct: we genuinely know its displacement label less
   well.
5. My quick correlation metric is a floor, not a ceiling — the TCN's learned
   filters and real spectral features will extract signal a 1 s rolling-std
   misses. `weak` sequences get re-evaluated once the model exists; the tiers
   are advisory metadata in the split manifest, not a hard delete.
