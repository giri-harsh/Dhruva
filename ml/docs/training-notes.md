# ANCHOR-Net training — findings & decisions

## The signal in IO-VNBD is real but modest (measured 2026-09-03)

The "speed from vibration texture" thesis relies on IMU energy that partly lives
above 10 Hz — and IO-VNBD's phone stream is 10 Hz native (PRD §6.1 says so). So
on IO-VNBD the achievable window-level speed accuracy is bounded:

| Predictor on the `test_id`-style held-out route | val speed RMSE |
|---|---|
| predict the training mean | 7.33 m/s |
| linear regression on 12 hand features | 6.92 |
| gradient boosting on the same hand features | 6.21 |
| **ANCHOR-Net TCN (MSE)** | **~5.9** |
| ANCHOR-Net TCN (MSE warm-up → β-NLL) | see latest run |

Most predictive single features (corr with speed): `std(accel_z)` 0.54,
`mean|gyro_z|` 0.51, `ptp(accel_z)` 0.51 — suspension + road-roughness response,
not high-frequency texture. `use`-tier sequences carry ~2.4× the correlation of
`weak`-tier (0.43 vs 0.18) — hence the `weak` down-weight in the loss.

**Implication:** window RMSE of ~5.9 m/s on a ~15 m/s mean sounds poor, but the
error is near-zero-mean and roughly independent across windows, so integrated
displacement over a 30–180 s outage averages down. The Week-5 gate metric is
drift-%, not window RMSE. And this is the reason the two-stage GNSS-pretrain on
the 58 h unsynchronised phone data matters (R-02 in the verification doc).

## Gaussian NLL from scratch is degenerate — use MSE warm-up

A 4-way probe (`use`-only, 10 epochs, lr 5e-4), val speed RMSE:

| objective | behaviour |
|---|---|
| MSE only | → 5.5 by ep4, stable ~5.9 |
| Gaussian NLL, no label σ | → 7.5 then **diverges** to 7.7 (variance head runs away) |
| Gaussian NLL, fixed σ=0.3 | → 7.6, same divergence |
| Gaussian NLL, large label σ (~3 m/s) | slow, then → **5.0** at ep5, then drifts up |

Two lessons:
1. Jointly optimising mean + variance from random init lets the variance head
   inflate σ and starve the mean gradient (`∂NLL/∂μ = (μ−y)/eff_var`). A large
   `eff_var` floor accidentally stabilises it, which is *why the earlier
   over-sized `label_sigma` "worked"* — it was load-bearing for the wrong reason.
2. The fix is explicit: **pure MSE for `warmup_epochs` (8), then β-NLL**
   (Seitzer et al. 2022 — `L = detach(eff_var**0.5) · NLL_per_sample`), which
   reduces the NLL's leverage on the mean while the variance still calibrates.
   `label_sigma` is then set to its *physically correct* small value (~0.25 m/s
   median — the labels really are that good) and is NOT relied on as a
   regulariser.

## label_sigma composition (ml/anchor/data/labels.py)

Four terms in quadrature × usability multiplier:
- `σ_wheelcan` wheel-integrated vs CAN indicated-speed disagreement over the window
- `σ_gnss` per-sequence RMS of (r·ω − VBOX speed) on clean straight stretches
  (≈ 0.17 m/s → the dominant term; VBOX GPS is metre-class)
- `σ_sync` `τ·|Δspeed across window|` (τ = 0.3 s) + per-seq `(1−sync_speed_corr)·2 m`
  — **not** `τ·speed` (that earlier form reached 3–7 m/s and swamped the label)
- `× 1.0 / 1.4 / 3.0` for usability `use` / `weak` / `drop`

Result: median label_sigma ≈ 0.21 m/s speed, p95 ≈ 0.49, which matches the
measured label-vs-VBOX residual (std 0.21 m/s). Labels are trustworthy;
uncertainty the variance head must explain is almost entirely epistemic.

## Config ablation (1 seed, 28 epochs, val = Vtb)

| config | val speed RMSE | converged | val bias |
|---|---|---|---|
| use-only, no aug | 5.61 | ep26 | −0.18 |
| **use + weak(0.4), no aug** | **5.47** | **ep11** | +0.01 |
| use-only, aug (7°) | 5.65 | ep26 | −0.26 |
| use + weak(0.4), aug (7°) | 5.68 | ep11 | +0.23 |

- **`weak`-tier data at 0.4 weight helps** and roughly halves epochs-to-converge
  (more data beats cleaner data here). Kept.
- **Heavy augmentation slightly hurts** on already-frame-aligned input — the
  synthetic rotations don't match a real deployment failure mode. Reduced to a
  light regulariser (3° rotation + noise, p 0.35); full strength is the
  augmentation ablation row.
- The earlier "stuck at 6.5, +2.5 bias" run was the **speed-decile
  re-weighting** (`WeightedRandomSampler`, PRD §6.6) at full strength distorting
  the training speed distribution. Softened: `weights ** 0.5`, ratio capped 4:1.

All configs beat predict-the-mean (7.33) by ~25% and the GBR hand-feature
ceiling (6.21). Headline 5-seed run: `ml/train/runs/week3_headAB/`.

## Perf note

`torch` default 12 threads oversubscribes this box (conv on tiny [B,40,20]
tensors) — `TrainConfig.num_threads = 5`, batch 512. ~10–13 s/epoch on
`use`+`weak` train (~52 k windows).
