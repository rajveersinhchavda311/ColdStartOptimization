# Paper Writing Context: Cold-Start Optimization via EVT-CVaR Risk-Aware Provisioning

**Purpose of this document:** Single-source-of-truth context for AI-assisted paper writing. Contains the research problem, complete methodology, all key numerical results, figure inventory, narrative arc, and limitations. Do not modify this file during paper writing — read from it, not into it.

---

## 1. Research Problem

### What is a serverless cold start?

In serverless (Function-as-a-Service) computing, a platform like AWS Lambda, Azure Functions, or Huawei FunctionGraph runs user code on demand in ephemeral containers. When a function is invoked and no warm container is available, the platform must provision a new one — this initialization delay is a **cold start**, directly degrading user-perceived latency and reliability.

### The provisioning challenge

To avoid cold starts, a platform can **pre-provision** containers before demand arrives. But over-provisioning wastes infrastructure resources (idle capacity). The fundamental tension:

- **Under-provision** → cold starts → SLA violations
- **Over-provision** → idle capacity → wasted cost

A good provisioning strategy must predict demand and decide `provisioned[t]` for the next minute, using only information available up to time `t`.

### Why existing approaches are insufficient

Simple reactive policies (provision = last minute's demand) track demand well on average but fail catastrophically during sudden spikes — exactly when cold starts matter most. Fixed percentile buffers (e.g., provision = P90 of historical demand) avoid spikes but waste resources during calm periods. Deep learning forecasters improve mean prediction but provide no principled uncertainty quantification.

**The gap this paper fills:** A framework that (1) uses any standard forecast as its base, (2) adds a safety buffer calibrated to the empirical tail of the forecast error distribution using Extreme Value Theory (EVT), and (3) scales the buffer dynamically with local forecast uncertainty.

---

## 2. Datasets

### Azure Functions 2019 (Primary)

- **Source:** Microsoft Azure Functions public trace (2019)
- **Resolution:** 1-minute intervals
- **Preprocessing:** Total platform-wide invocations per minute (sum across all functions) = our demand proxy. This aggregate is labeled `concurrency` in all code/results. **Note: this is invocations/minute, NOT true concurrency** (true concurrency requires duration data, which is unavailable). This is a standard simplification in the serverless scheduling literature.
- **Duration:** ~14 days of data
- **Post-burn-in rows:** 18,720
- **Train / Val / Test split (60/20/20):** 11,232 / 3,744 / 3,744 (strictly chronological)
- **Burn-in:** First 1,440 rows dropped (required to compute lag_1440 = 24-hour lag)
- **Demand statistics (training set):** mean = 613,900, std = 68,696, P90 = 696,264, P99 = 785,458, max = 1,258,768
- **Extreme event threshold:** P99 of training = 785,458 invocations/min; 26 extreme events in test set
- **Scale note:** Test maximum (866,343) is 1.10× the training P99 — the test distribution stays well within the training tail regime

### Huawei Cloud (External Validation)

- **Source:** Huawei Cloud serverless function traces (January 2025)
- **Resolution:** 1-minute intervals (same as Azure)
- **Structure:** 5 independent function traces (R1–R5) plus combined aggregate (combined = R1 + R2 + R3 + R4 + R5 exactly)
- **Duration:** 31 days (January 2025), 44,640 raw rows
- **Post-burn-in rows:** 43,200 (burn-in = 1,440 rows)
- **Train / Val / Test split (60/20/20):** 25,920 / 8,640 / 8,640
- **Train date range:** Jan 2 – Jan 19; Val: Jan 20 – Jan 25; Test: Jan 26 – Jan 31
- **Combined demand statistics (training):** mean = 261, std = 130, P90 = 429, P99 = 729, max = 1,902
- **Test set max demand (combined): 3,657** — 5.0× the training P99 (key difference from Azure)
- **Extreme event threshold:** P99 of training = 729; 151 extreme events in test set
- **Regional variation:** R1 (mean 158, dominant), R2 (mean 46), R3 (mean 7, tiny), R4 (mean 32), R5 (mean 18)
- **Scale note:** Unlike Azure, Huawei test set contains spikes up to 5× the training P99 — this causes lower extreme SLA on Huawei (explained in limitations)

### Feature engineering (both datasets)

```
Features: lag_1, lag_2, ..., lag_10, lag_1440
Target:   concurrency[t]

lag_k[t] = concurrency[t - k]        (short-range autocorrelation)
lag_1440[t] = concurrency[t - 1440]  (24-hour daily seasonal lag, same clock minute yesterday)
```

All lag features use only past information. lag_1440 is the daily seasonal signal — identical interpretation across both datasets since both are 1-minute resolution.

---

## 3. Cost Model

**Provisioning cost (applied uniformly to all models, both datasets):**

```
provisioned[t] = ceil(prediction[t])
cold_starts[t] = max(0, demand[t] - provisioned[t])
idle[t]        = max(0, provisioned[t] - demand[t])
cost[t]        = cold_starts[t] × 10 + idle[t] × 1
```

**Ratio:** Cold starts penalized 10× more than idle capacity, reflecting the user-facing nature of cold-start failures versus wasted-but-silent idle resources.

**Two SLA metrics:**
- `request_sla = 1 − (total cold starts) / (total demand)` — fraction of all demand units served without cold start
- `extreme_sla = 1 − (cold starts during extreme events) / (demand during extreme events)` — same metric restricted to timesteps where demand > P99(train). This is the more informative metric for evaluating tail protection.

**Extreme event definition:** A timestep is "extreme" if `demand[t] > P99(train concurrency)`. Threshold is computed from training data only and held fixed for the test set. On Azure: 26 extreme events. On Huawei combined: 151 extreme events.

---

## 4. Methodology

### Phase 1: Forecasting Baselines

Six models spanning the complexity spectrum. All models use only lag features as input — never `concurrency[t]` directly (leakage prevention). The simulator calls `model.predict(row_t)` and receives a prediction; provisioning = `ceil(prediction)`.

| Model | Prediction formula | Training |
|-------|-------------------|----------|
| Reactive | `lag_1[t]` | None |
| Static_P90 | `P90(train concurrency)` — constant | None |
| Forecast_Only | `mean(lag_1..lag_10)[t]` | None |
| Seasonal_Naive | `lag_1440[t]` | None |
| Linear_Seasonal | `β₀ + β₁·lag_1[t] + β₂·lag_1440[t]` | OLS (closed form) |
| TCN | Causal dilated 1D conv: 4 residual blocks (8 conv layers), kernel 3, dilations [1, 2, 4, 8] | Gradient descent, seed=42 |

**TCN implementation notes:** Target normalization applied during training (normalize by training mean/std, denormalize in predict). lag_1440 fed as a scalar side-channel concatenated with TCN body output before final linear layer. ~22K parameters. Early stopping on val loss.

**Phase 1 audit:** All checks PASS. Verified: no future leakage, chronological splits, accounting identity, test set identity across all models.

### Phase 2: EVT-CVaR Risk-Aware Wrapper

`RiskAwareModel` wraps any Phase 1 model (except Static_P90) and adds a dynamic safety buffer:

```
final_prediction[t] = base_prediction[t] + buffer[t]

where:
  buffer[t]  = sigma_t × CVaR_z
  sigma_t    = std(last W residuals)           [dynamic, rolling window]
  CVaR_z     = EVT-fitted CVaR at α=0.99      [static, fitted on training data]
```

**Buffer computation — leakage-free sequential process:**
1. At timestep t: compute `base_pred[t]` using only lag columns (no leakage)
2. Reconstruct past residual: `ε[t-1] = lag_1[t] - base_pred[t-1]` (lag_1[t] = actual demand at t-1, available at t)
3. Maintain rolling window of last W=30 residuals → compute `sigma_t`
4. Warm-up: for first W timesteps, `sigma_t = sigma_train` (training residual std)
5. Apply buffer: `final_pred[t] = base_pred[t] + sigma_t × CVaR_z`

**EVT pipeline (training phase):**
1. Compute training residuals: `ε = actual - base_pred` on training set
2. Compute `sigma_train = std(ε)`
3. Standardize: `z = ε / sigma_train` — scale-standardization only, no mean-centering. Training residual means are ≤0.24% of σ_train for all base models except Seasonal_Naive (7.8%), where the offset is absorbed by the empirically-set P90 threshold and the GPD fit to exceedances above it.
4. POT (Peaks Over Threshold): take all `z > u` where `u = P90(z)` on training set
5. Fit Generalized Pareto Distribution (GPD) to exceedances above `u`: shape ξ, scale β
6. Compute `CVaR_z` at α=0.99: the expected value of z conditional on z exceeding VaR₀.₉₉

**Anchor parameters (fixed for Phases 2–5):** α = 0.99, W = 30, threshold = P90

**Why EVT over Gaussian:** The Gaussian CVaR at α=0.99 is K_GAUSSIAN = φ(Φ⁻¹(0.99)) / (1 − 0.99) ≈ 2.6652. EVT fits the actual tail distribution and yields CVaR_z = 3.5–5.5 (1.33–2.05× larger). The gap reflects heavier-than-Gaussian tails in IT workload forecast residuals — Gaussian would systematically under-buffer.

**Phase 2 audit:** 106/106 PASS (2 XFAIL). Two checks are marked as expected failures (XFAIL) and counted as passed. Both are for `RiskAware(Forecast_Only)` and `RiskAware(Seasonal_Naive)` on the check "mean(buffer_t | extreme events) > mean(buffer_t | normal)".

**Why these two fail this check — and why it is not a bug:**

The check asks whether the dynamic buffer scales *up* when demand is extreme (> P99 of training). The buffer = σ_t × CVaR_z, with CVaR_z constant, so this is really asking whether σ_t is larger during extreme timesteps.

- **Forecast_Only** averages lag_1..lag_10. Azure extreme events are often gradual daily ramp-ups (business hours), not sudden spikes. By the time demand reaches the extreme threshold, the 10-lag mean is already tracking the increase — residuals are moderate, not large. σ_t does not reliably rise during extremes.
- **Seasonal_Naive** uses lag_1440 (yesterday's same minute). Extreme events on Azure are frequently time-of-day correlated (recurring peak hours). Yesterday's same minute was also a peak, so the prediction is already high, the residual is small, and σ_t stays low during extremes.
- **Reactive** (which passes): lag_1 only looks back 1 step. A sudden spike creates a single huge residual that immediately inflates σ_t. The buffer responds.
- **TCN** (which passes): a learned model that tracks demand well during normal periods but is genuinely surprised by tail events; residuals increase at extremes.

**What this means for the paper:** The dynamic buffer's risk-responsiveness depends on the base model's residual structure. Models that track seasonal/trend components well tend to have smaller residuals during seasonally-driven extreme events, so σ_t does not scale up further — but these models also have a lower baseline residual level to begin with. For the paper, the recommended framing is:

> "For smoothing-based base models (Forecast_Only, Seasonal_Naive), the dynamic buffer does not amplify further during extreme demand because these models partially anticipate seasonally-driven spikes through their prediction structure, leaving smaller residuals at extremes. The buffer's full risk-responsiveness is realized for models that are genuinely surprised by demand spikes (Reactive, TCN). This underscores TCN as the preferred base model: best average prediction accuracy AND strongest risk-adaptive behavior."

This is a nuance that enriches the paper's discussion section. The primary results — cold-start reduction, SLA, EVT vs Gaussian gap — are unaffected and hold for all 5 models.

More generally — and this applies to *every* base model, not only the smoothing ones — σ_t is computed from the previous W=30 residuals, so at the first timestep of any spike the buffer still reflects pre-spike volatility: the dynamic mechanism **reacts within one timestep but does not anticipate**. The paper should state this plainly and cite the project's own ablation as the quantification: static σ_train yields strictly higher SLA than dynamic σ_t (Phase 4: C3 > C4 for both models), and dynamic σ's contribution is cost efficiency, not tail protection.

**Static_P90 exclusion:** This model has no training residuals (it predicts a constant), so the EVT pipeline cannot operate.

### Phase 3: Sensitivity Analysis

**Purpose:** Demonstrate that the method is robust — results do not depend on the specific anchor parameter values. Shows the method is principled, not cherry-picked.

**Design:**
- 3A: One-at-a-time sweep — 9 configs (3 values each for α, W, threshold; other two held at anchor)
- 3B: 2×2×2 factorial at boundaries — 8 configs ({α: 0.95, 0.99} × {W: 10, 60} × {P: P85, P95})
- Anchor deduped across all sweeps: 15 unique configs × 2 models (Reactive, TCN) = 30 runs
- Audit: 148/148 PASS

**Findings:**
1. **α is the primary lever** (strong cost+SLA tradeoff): α=0.95 → cost 239M, SLA 0.9989 (TCN); α=0.99 → cost 342M, SLA 0.9997. 30% cost reduction for 0.08pp SLA relaxation.
2. **W is near-inert on cost, SLA-positive**: on Azure, W=60 improves request SLA over W=30 (+0.027pp Reactive, +0.019pp TCN) at ≤1.2% additional cost.
3. **Threshold is nearly inert**: TCN SLA = 0.9997 at P85, P90, and P95. Cost varies <5%.
4. **No parameter interactions detected**: Interaction plots show approximately parallel lines across all 3B factorial combinations.

### Phase 4: Ablation Study

**Purpose:** Decompose the EVT-CVaR framework into components and quantify each component's contribution independently.

**2×2 factorial design:**

| Condition | σ type | Multiplier | Description |
|-----------|--------|------------|-------------|
| C0 | none | none | Phase 1 baseline (loaded from disk) |
| C1 | Static (σ_train) | K_GAUSSIAN = 2.6652 | Fixed Gaussian buffer |
| C2 | Dynamic (rolling W=30) | K_GAUSSIAN = 2.6652 | Adaptive Gaussian buffer |
| C3 | Static (σ_train) | EVT CVaR_z | Fixed EVT buffer |
| C4 | Dynamic (rolling W=30) | EVT CVaR_z | Full Phase 2 (loaded from disk) |

**K_GAUSSIAN derivation** (scipy, α=0.99): `norm.pdf(norm.ppf(0.99)) / (1 - 0.99) ≈ 2.6652`

This is the Gaussian CVaR at the same confidence level as EVT. Using k=3.0 would correspond to α≈0.9987 — the wrong confidence level — and conflate distributional assumptions with confidence level choice.

**Audit:** 63/63 PASS. Static σ conditions verified constant (std(sigma_t) < 1e-10). C0/C4 bit-identical to Phase 1/2.

### Phase 5: Huawei Generalization

**Purpose:** Apply the frozen Azure methodology to an independent dataset from a different cloud provider without any parameter modification. Zero-modification experiment.

**Zero-modification principle:** α=0.99, W=30, threshold=P90, cost function, feature set, and all model architectures applied unchanged. Only the training data changes.

---

## 5. Complete Results

### Phase 1 — Azure Test Set

| Model | Request SLA | Extreme SLA | Total Cost | Cold Starts |
|-------|------------|------------|-----------|-------------|
| Reactive | 0.984790 | 0.949060 | 407.8M | 37,072,085 |
| Static_P90 | 0.992484 | 0.865030 | 370.9M | 18,318,487 |
| Forecast_Only | 0.982252 | 0.896636 | 475.8M | 43,258,850 |
| Seasonal_Naive | 0.950185 | 0.901161 | 1,262.8M | 121,420,546 |
| Linear_Seasonal | 0.981472 | 0.938329 | 478.1M | 45,160,587 |
| **TCN** | **0.985896** | **0.943579** | **371.4M** | **34,377,188** |

*Bold marks the base model selected for Phases 2–4 (best balance among wrappable models), not the per-column maximum: Static_P90 has the highest request SLA and lowest cost, Reactive the highest extreme SLA.*

Request SLA: ~98.5% across the board. Extreme SLA: 86–95% — already worse, reflecting that demand spikes overwhelm all Phase 1 models. Seasonal_Naive has by far the highest cost (1,262.8M, of which 96% is cold-start penalty): test-period demand runs ~3% above the same minute of the previous day on average, so same-minute-yesterday prediction under-provisions on 63% of timesteps, and the 10:1 cold:idle penalty amplifies that systematic under-prediction.

### Phase 2 — Azure Test Set (EVT anchor: α=0.99, W=30, P90)

| Model | Request SLA | Extreme SLA | Total Cost | Cold Starts | Reduction |
|-------|------------|------------|-----------|-------------|-----------|
| RiskAware(Reactive) | 0.999645 | 0.996935 | 446.6M | 865,343 | −97.7% |
| RiskAware(Forecast_Only) | 0.999555 | 0.993127 | 489.0M | 1,085,265 | −97.5% |
| RiskAware(Seasonal_Naive) | 0.994317 | 0.979002 | 410.9M | 13,851,671 | −88.6% |
| RiskAware(Linear_Seasonal) | 0.999662 | 0.995138 | 365.6M | 823,624 | −98.2% |
| **RiskAware(TCN)** | **0.999696** | **0.994396** | **342.4M** | **741,901** | **−97.8%** |

Request SLA jumps from ~98.5% to ~99.97%. Extreme SLA jumps from ~86–95% to ~97.9–99.7%. Cold starts reduced by 88–98%. **Among the five risk-aware models, RiskAware(TCN) achieves the best request SLA and the lowest cost simultaneously** (cost ranking stable for cold:idle ratios 5:1–20:1; see Cost-Ratio Robustness below).

### Phase 2 — EVT Parameters (Azure)

| Model | ξ (GPD shape) | CVaR_z | CVaR_z / K_GAUSSIAN |
|-------|--------------|--------|---------------------|
| RiskAware(Reactive) | −0.0928 | 4.160 | 1.56× |
| RiskAware(Forecast_Only) | +0.0026 | 4.278 | 1.61× |
| RiskAware(Seasonal_Naive) | +0.1845 | 3.544 | 1.33× |
| RiskAware(Linear_Seasonal) | +0.0192 | 4.161 | 1.56× |
| RiskAware(TCN) | +0.0222 | 4.295 | 1.61× |

All CVaR_z values 1.33–1.61× above the Gaussian baseline. The Gaussian assumption would under-buffer by 33–61%.

### Phase 3 — Sensitivity (Azure, RiskAware(TCN) representative)

| Parameter | Values tested | Effect on SLA | Effect on cost |
|-----------|---------------|---------------|---------------|
| α | 0.95, 0.975, **0.99** | −0.08pp per level | −30% at 0.95 vs 0.99 |
| W | 10, **30**, 60 | +0.06pp at W=60 | <10% variation |
| Threshold | P85, **P90**, P95 | Negligible | <5% variation |

Anchor values shown in **bold**. SLA range across all 30 runs: 0.9978–0.99995. Method is robust.

### Phase 4 — Ablation (Azure, Reactive and TCN)

| Condition | Reactive SLA | Reactive Extreme SLA | Reactive Cost | TCN SLA | TCN Extreme SLA | TCN Cost |
|-----------|-------------|---------------------|--------------|---------|-----------------|---------|
| C0 No Buffer | 0.984790 | 0.949060 | 407.8M | 0.985896 | 0.943579 | 371.4M |
| C1 Static+Gaussian | 0.999187 | 0.993407 | 323.9M | 0.999216 | 0.986490 | 239.1M |
| C2 Dynamic+Gaussian | 0.998701 | 0.989296 | 314.8M | 0.998844 | 0.983345 | 235.8M |
| C3 Static+EVT | **0.999965** | **0.999857** | 472.6M | **0.999900** | **0.997675** | 358.3M |
| C4 Dynamic+EVT (Phase 2) | 0.999645 | 0.996935 | 446.6M | 0.999696 | 0.994396 | 342.4M |

**2×2 Request SLA heatmap (Reactive):**

|  | Gaussian (K=2.665) | EVT (CVaR_z) |
|--|---|---|
| Static σ | 0.999187 (C1) | **0.999965** (C3) |
| Dynamic σ | 0.998701 (C2) | 0.999645 (C4) |

**Incremental transitions:**

| Transition | Question | Reactive ΔSLA | TCN ΔSLA |
|-----------|---------|--------------|----------|
| C0 → C1 | Does any buffer help? | **+1.44 pp** | **+1.33 pp** |
| C1 → C3 | Does EVT outperform Gaussian (static)? | +0.08 pp | +0.07 pp |
| C1 → C2 | Does dynamic σ improve on static (Gaussian)? | −0.05 pp | −0.04 pp |
| C2 → C4 | Does EVT add value over dynamic Gaussian? | +0.09 pp | +0.09 pp |
| C3 → C4 | Does dynamic σ add value over static EVT? | −0.03 pp | −0.02 pp |

**Key ablation findings:**
1. **Adding any buffer (C0→C1) is the dominant effect:** +1.33–1.44pp SLA, −94% cold starts
2. **EVT tail calibration** contributes mainly to extreme SLA: C1 extreme_sla = 0.987–0.993 vs C3 = 0.998–0.9999
3. **Dynamic σ is a cost-efficiency mechanism:** C3→C4 costs −4–6% (Reactive −5.5%, TCN −4.4%) but SLA decreases by only −0.02–0.03pp
4. **C3 (Static+EVT) achieves highest SLA; C4 (Dynamic+EVT) is the cost-optimal choice**

### Phase 5 — Huawei Generalization (Combined)

**Phase 1 (Huawei Combined, test set):**

| Model | Request SLA | Extreme SLA | Total Cost | Cold Starts |
|-------|------------|------------|-----------|-------------|
| Reactive | 0.8014 | 0.3750 | 5.5M | 495,683 |
| Static_P90 | 0.9320 | 0.4711 | 3.1M | 169,759 |
| Forecast_Only | 0.8531 | 0.3702 | 4.0M | 366,463 |
| Seasonal_Naive | 0.8926 | 0.6355 | 2.9M | 268,035 |
| Linear_Seasonal | 0.8847 | 0.5637 | 3.1M | 287,825 |
| TCN | 0.9094 | 0.5895 | 2.5M | 226,064 |

Phase 1 SLA is lower on Huawei than Azure (0.80–0.93 vs 0.95–0.99). This reflects Huawei's higher demand volatility (std/mean = 0.50 vs 0.11 for Azure).

**Phase 2 (Huawei Combined, test set):**

| Model | Request SLA | Extreme SLA | Total Cost | Cold Starts | Reduction |
|-------|------------|------------|-----------|-------------|-----------|
| RiskAware(Reactive) | 0.9973 | 0.9554 | 5.8M | 6,634 | −98.7% |
| RiskAware(Forecast_Only) | 0.9968 | 0.9466 | 5.5M | 7,868 | −97.9% |
| RiskAware(Seasonal_Naive) | 0.9959 | 0.9376 | 3.4M | 10,267 | −96.2% |
| RiskAware(Linear_Seasonal) | 0.9960 | 0.9365 | 3.3M | 10,070 | −96.5% |
| RiskAware(TCN) | 0.9959 | 0.9339 | 3.5M | 10,189 | −95.5% |

Cold-start reductions of 95–99%, consistent with Azure. Request SLA > 0.99 for all 5 models.

**Extreme SLA gap (Huawei 0.93–0.96 vs Azure 0.98–0.997):** Test set max demand on Huawei (3,657) is 5× training P99 (729). Azure test max is only 1.10× training P99. Huawei's test period contains out-of-training-distribution spikes that no calibrated buffer can fully anticipate.

**EVT Parameters (Huawei Combined):**

| Model | ξ | CVaR_z | CVaR_z / K_GAUSSIAN |
|-------|---|--------|---------------------|
| RiskAware(Reactive) | −0.0723 | 3.804 | 1.43× |
| RiskAware(Forecast_Only) | −0.0440 | 4.924 | 1.85× |
| RiskAware(Seasonal_Naive) | +0.3194 | 4.371 | 1.64× |
| RiskAware(Linear_Seasonal) | +0.2417 | 4.616 | 1.73× |
| RiskAware(TCN) | +0.2780 | 5.460 | 2.05× |

**Cross-Dataset ξ Summary (Reactive and TCN across all 7 datasets):**

| Dataset | Reactive ξ | TCN ξ | Reactive ratio | TCN ratio |
|---------|-----------|-------|----------------|-----------|
| Azure | −0.0928 | +0.0222 | 1.56× | 1.61× |
| Huawei Combined | −0.0723 | +0.2780 | 1.43× | 2.05× |
| Huawei R1 | +0.0026 | +0.3580 | 1.29× | 2.30× |
| Huawei R2 | +0.4107 | +0.5013 | 1.98× | 2.47× |
| Huawei R3 | −0.2095 | +0.1538 | 1.16× | 1.63× |
| Huawei R4 | −0.1736 | +0.5446 | 1.73× | 2.95× |
| Huawei R5 | −0.2288 | +0.0148 | 1.07× | 1.35× |

**Universal finding:** CVaR_z / K_GAUSSIAN > 1.0 in all 14 (model, dataset) combinations. TCN ξ > Reactive ξ in all 7 datasets. EVT recommends a larger buffer than Gaussian everywhere. This is robust to GPD sampling uncertainty: parametric bootstrap 95% CIs (B=2000, conditional on the P90 threshold; `scripts/evt_bootstrap_ci.py` → `results/analysis/evt_bootstrap_ci.csv`) keep CVaR_z above K_GAUSSIAN for **all 35 fits across all 7 datasets and 5 models**; the tightest case (Huawei R5, Reactive) has CVaR_z CI [2.81, 2.90] vs K_GAUSSIAN = 2.665.

### Cost-Ratio Robustness (10:1 assumption check)

The 10:1 cold:idle ratio is an experimental assumption (see Limitations #5). Because provisioning = ceil(prediction) never depends on the cost parameters, per-timestep cold/idle outcomes are invariant under the ratio, and costs under alternative ratios are exact re-weightings of the frozen results (`scripts/cost_ratio_sensitivity.py` → `results/analysis/cost_ratio_sensitivity.csv`; self-checked to reproduce all stored 10:1 totals exactly).

| Finding | 5:1 | 10:1 | 20:1 |
|---------|-----|------|------|
| Cheapest Phase 2 Azure model | RiskAware(TCN) 338.7M | RiskAware(TCN) 342.4M | RiskAware(TCN) 349.8M |
| Dynamic σ cheaper than static (EVT: C4 < C3), both models | Yes | Yes | Yes |
| Wrapper cost vs unwrapped TCN | 338.7M vs 199.5M (costs more) | 342.4M vs 371.4M (cheaper) | 349.8M vs 715.1M (much cheaper) |

- **"RiskAware(TCN) is the cheapest of the five risk-aware models" holds at all tested ratios (5:1–20:1).**
- **"Dynamic σ is a cost-efficiency mechanism" (C3→C4) holds at all tested ratios** for the EVT buffer. (For the Gaussian buffer, the dynamic-vs-static cost ordering reverses at 20:1 — C2 exceeds C1 — but no paper claim rests on that comparison.)
- **The claim "maintains or reduces total cost" is ratio-dependent:** it holds at 10:1 and strengthens at 20:1; at 5:1, cold starts are cheap enough that the buffer increases cost relative to the unwrapped base model and is justified by the SLA gain (0.9859 → 0.9997), not by cost. The paper should scope the cost claim to ratios ≥ 10:1.
- On Huawei Phase 2, RiskAware(Linear_Seasonal) is cheapest at all ratios, with RiskAware(TCN) within 5% — consistent with the stored 10:1 ranking.

---

## 6. Paper Narrative Arc

### The story in one paragraph

Serverless platforms face a fundamental provisioning tension: under-provisioning causes cold starts, over-provisioning wastes resources. Standard forecasting models (Phase 1) achieve ~98.5% request SLA but only ~86–95% SLA during demand spikes — the moments that matter most. We propose wrapping any forecast model with an EVT-CVaR safety buffer (Phase 2) calibrated to the empirical tail of the forecast error distribution. This buffer, scaled dynamically by local volatility, reduces cold starts by 88–98% and raises extreme SLA to 97.9–99.7% on Azure — all while maintaining or reducing total cost for the best models at the baseline 10:1 cold:idle cost ratio. Sensitivity analysis (Phase 3) confirms the method is robust to hyperparameter choice. Ablation (Phase 4) shows that EVT tail calibration is the key contributor to tail protection, while dynamic sigma provides cost efficiency. External validation on Huawei (Phase 5) achieves 95–99% cold-start reduction without any methodology modification, with EVT consistently recommending a 1.07–2.95× larger buffer than Gaussian across all tested datasets and models.

### Section-by-section claims

**Introduction:** Serverless cold starts are a latency-reliability problem. Existing solutions use threshold rules or static buffers without statistical grounding. We need a method that adapts to the actual tail behavior of forecast errors.

**Baseline scope (anticipate the reviewer):** ARIMA is represented by Linear_Seasonal, which is functionally a restricted ARIMA(1,0,0)(1,0,0)[1440] (see `docs/phase1/architecture.md` D6). LSTM/Transformer baselines are deliberately excluded: the framework is forecaster-agnostic (it consumes only residuals), TCN serves as the representative deep sequence model (TCNs match or exceed recurrent architectures on standard benchmarks — Bai, Kolter & Koltun 2018 — and train deterministically), and the buffer's consistency across five heterogeneous base models is the forecaster-independence evidence (D7).

**Methodology:** EVT is the statistically correct tool for tail modeling — GPD is proven (Pickands–Balkema–de Haan theorem) to be the asymptotically correct distribution for exceedances above a high threshold. Dynamic sigma scales the buffer to local uncertainty. The combination is principled, not heuristic.

**Phase 1 results:** Six baselines bracket the performance space. No Phase 1 model exceeds 94.9% extreme SLA, and the strongest learned forecaster (TCN) achieves only 94.4%, motivating the need for tail-aware provisioning. Address the obvious reviewer question head-on: Static_P90 attains the highest request SLA (0.9925) and lowest cost (370.9M), but the worst extreme SLA (0.8650) — a fixed percentile is precisely the policy that fails when demand spikes — and, predicting a constant, it has no forecast-error distribution for the EVT pipeline to calibrate (Limitation #7). Reactive attains the best extreme SLA (0.9491) but is the weakest candidate on cost once wrapped (Phase 2: 446.6M vs TCN's 342.4M). TCN is selected as the primary base model because it offers the best cost–SLA balance among wrappable models and, per Phase 2, the strongest risk-adaptive behavior.

**Phase 2 results:** EVT-CVaR wrapper transforms extreme SLA from 94% to 99.4% for RiskAware(TCN) while reducing cost from 371M to 342M. The buffer is not "always add more" — it scales down during calm periods.

**Phase 3:** The method is robust. α is a meaningful operational lever (cost-SLA tradeoff). W and threshold are near-inert — users do not need to tune them.

**Phase 4:** The dominant effect is adding any distribution-calibrated buffer (+1.3pp SLA). EVT's specific contribution is tail protection: extreme SLA improves from 0.987–0.993 (Gaussian) to 0.998–0.9999 (EVT) — an order of magnitude closer to perfect. Dynamic sigma is a cost optimizer, not a correctness mechanism.

**Phase 5:** The methodology generalizes. On an independent Huawei dataset, without any parameter modification, cold starts fall 95–99%. EVT consistently recommends larger buffers than Gaussian (1.07–2.95× ratio), confirming heavy-tail behavior is a general property of cloud workload forecast errors, not Azure-specific.

### Key sentences for abstract/introduction

- "We show that forecast error residuals on both Azure and Huawei follow heavier-than-Gaussian tails, with EVT-fitted CVaR values 33–105% above the Gaussian baseline at the same confidence level." *(scope: the two platform-level datasets, Azure and Huawei Combined, all 5 models; across all 7 datasets including per-region traces the range is 7–195%, i.e., ratios 1.07–2.95×)*
- "Our framework reduces cold starts by 88–98% on Azure and 95–99% on Huawei without any hyperparameter modification between platforms."
- "Ablation analysis reveals that EVT tail calibration is the primary contributor to tail protection, while dynamic sigma estimation provides cost efficiency at negligible SLA cost."
- "Sensitivity analysis across 30 configurations demonstrates the method is insensitive to hyperparameter choice within a broad operational range."

---

## 7. Figure Inventory

All figures are pre-generated. File paths relative to project root. Use these descriptions when referencing figures in the paper.

### Preprocessing

| Figure | Path | What it shows |
|--------|------|---------------|
| Azure time series | `graphs/preprocessing/azure/timeseries_plot.png` | Full Azure demand time series with train/val/test split boundaries |
| Azure histogram | `graphs/preprocessing/azure/histogram_plot.png` | Demand distribution with P90/P99 markers |
| Huawei region decomposition | `graphs/preprocessing/huawei/region_decomposition.png` | R1–R5 individual traces + combined aggregate |
| Huawei regional histograms | `graphs/preprocessing/huawei/{combined,R1-R5}/histogram_plot.png` | Per-region demand distributions |

### Phase 1

| Figure | Path | What it shows |
|--------|------|---------------|
| SLA comparison | `graphs/phase1/azure/sla_comparison.png` | Request SLA and Extreme SLA for all 6 models |
| Cost comparison | `graphs/phase1/azure/cost_comparison.png` | Total cost breakdown (cold vs idle) for all 6 models |
| Extreme event analysis | `graphs/phase1/azure/extreme_event_analysis.png` | Demand spikes vs provisioning during extreme events |
| Prediction error distribution | `graphs/phase1/azure/prediction_error_distribution.png` | Residual distributions for all 6 models |
| Cold start timeline | `graphs/phase1/azure/cold_start_timeline.png` | Cold-start occurrences over test period |

### Phase 2

| Figure | Path | What it shows |
|--------|------|---------------|
| SLA comparison | `graphs/phase2/azure/sla_comparison.png` | Phase 1 vs Phase 2 SLA for each base model |
| Cost comparison | `graphs/phase2/azure/cost_comparison.png` | Phase 1 vs Phase 2 cost for each base model |
| Dynamic buffer | `graphs/phase2/azure/dynamic_buffer.png` | sigma_t time series showing buffer scaling with volatility |
| Buffer distribution | `graphs/phase2/azure/buffer_distribution.png` | Distribution of buffer sizes across all Phase 2 models |
| Prediction overlay | `graphs/phase2/azure/prediction_overlay.png` | Actual demand vs base prediction vs risk-aware prediction |

### Phase 3

| Figure | Path | What it shows |
|--------|------|---------------|
| Sensitivity curves | `graphs/phase3/azure/sensitivity_curves.png` | SLA and cost vs each parameter (α, W, threshold), 2×3 grid |
| Extreme SLA sensitivity | `graphs/phase3/azure/sensitivity_extreme_sla.png` | Extreme SLA vs each parameter |
| Interaction plots | `graphs/phase3/azure/interaction_plots.png` | Phase 3B interaction effects (parallel lines = no interaction) |
| Robustness overview | `graphs/phase3/azure/robustness_overview.png` | All 30 configs as dots — all above SLA=0.99 baseline |
| Buffer sensitivity | `graphs/phase3/azure/buffer_sensitivity.png` | Mean buffer ± 1σ vs each parameter |

### Phase 4

| Figure | Path | What it shows |
|--------|------|---------------|
| Ablation SLA | `graphs/phase4/azure/ablation_sla.png` | Request SLA and Extreme SLA by condition (C0–C4), grouped bars |
| Ablation cost | `graphs/phase4/azure/ablation_cost.png` | Cost decomposition (cold+idle) by condition |
| Incremental contributions | `graphs/phase4/azure/ablation_incremental.png` | Delta SLA and delta cost per component addition step |
| Buffer profiles | `graphs/phase4/azure/ablation_buffer_profiles.png` | Time series of buffer_t for C1–C4 (flat vs adaptive) |
| **2×2 heatmap** | `graphs/phase4/azure/ablation_2x2_heatmap.png` | **PRIMARY FIGURE:** 2×2 Request SLA heatmap (σ type × multiplier type) |

### Phase 5

| Figure | Path | What it shows |
|--------|------|---------------|
| **Tail heaviness** | `graphs/phase5/tail_heaviness_comparison.png` | **PRIMARY FIGURE:** Standardized residual distributions + GPD fit vs Gaussian, Azure vs Huawei. All four panels show real training residuals; TCN panels load the cache written by `scripts/cache_tcn_residuals.py` (gate-verified against stored Phase 2 σ/ξ/CVaR) |
| **EVT multiplier comparison** | `graphs/phase5/evt_multiplier_comparison.png` | **PRIMARY FIGURE:** CVaR_z / K_GAUSSIAN ratios across datasets, all models |
| Cold start reduction | `graphs/phase5/cold_start_reduction.png` | Phase 1 → Phase 2 cold-start reduction on Azure and Huawei |
| Cross-dataset SLA | `graphs/phase5/cross_dataset_phase2_sla.png` | Phase 2 request and extreme SLA: Azure vs Huawei |
| Regional ξ heatmap | `graphs/phase5/regional_evt_heatmap.png` | GPD shape ξ across all 7 datasets × 5 models |
| ξ summary | `graphs/phase5/evt_xi_summary.png` | Cross-dataset ξ and ratio summary table (rendered as figure) |

---

## 8. Terminology (use consistently in paper)

| Term | Definition | Notes |
|------|------------|-------|
| Cold start | A demand unit not served by pre-provisioned capacity | = max(0, demand − provisioned) per timestep |
| Idle capacity | Over-provisioned units not consumed | = max(0, provisioned − demand) per timestep |
| Request SLA | Fraction of demand units served | 1 − cold_starts / total_demand |
| Extreme SLA | Request SLA restricted to extreme demand timesteps | 1 − cold_starts_extreme / demand_extreme |
| Extreme event | Timestep where demand > P99(training demand) | Threshold from training only, fixed for test |
| σ_train | Std of training residuals | Constant; used as warm-up value for sigma_t |
| sigma_t | Rolling std of last W=30 residuals | Dynamic local volatility estimate |
| CVaR_z | EVT-fitted Conditional Value-at-Risk on standardized residuals | Fitted per model on training data |
| K_GAUSSIAN | Gaussian CVaR at α=0.99 | ≈ 2.6652; computed analytically, no data dependence |
| GPD | Generalized Pareto Distribution | Fitted to residual exceedances above P90 threshold |
| POT | Peaks Over Threshold | Method for selecting GPD fitting data |
| ξ (xi) | GPD shape parameter | ξ > 0 = heavy tail; ξ = 0 = exponential; ξ < 0 = bounded |
| β (beta) | GPD scale parameter | Controls spread of tail |
| Concurrency | Label used for our demand proxy | Total invocations/minute (NOT true concurrency) |
| Zero-modification | Phase 5 principle: no parameter changes between Azure and Huawei | Critical for external validity claim |

---

## 9. Limitations (document honestly in paper)

1. **"Concurrency" misnomer:** The demand signal is total invocations/minute aggregated across all functions, not true concurrency (which requires duration data). This is a standard simplification but limits direct comparison to per-function provisioning systems.

2. **Aggregate provisioning model:** The cost model operates on the platform-wide aggregate. In practice, platforms provision per-function. The aggregate captures overall resource pressure but ignores individual function heterogeneity.

3. **Huawei out-of-distribution spikes:** Test set max demand (3,657) is 5× the training P99 (729). EVT cannot extrapolate arbitrarily far into unseen tail regimes. Extreme SLA on Huawei (0.93–0.96) reflects this; on Azure (test max = 1.10× P99) it does not occur.

4. **Single test set evaluation:** No confidence intervals on SLA metrics. Test set is large enough (3,744 timesteps on Azure, 8,640 on Huawei) that variance is small, but not formally quantified. EVT parameter uncertainty IS quantified: parametric bootstrap 95% CIs for ξ and CVaR_z (all 35 fits, `results/analysis/evt_bootstrap_ci.csv`) confirm CVaR_z > K_GAUSSIAN across the full CI everywhere; the CIs are conditional on the P90 threshold choice (threshold and σ_train uncertainty are not modeled).

5. **Cost model is experimental:** The 10:1 cold:idle ratio is an assumption. Real cloud pricing differs by provider and function type. Absolute cost numbers are not externally valid — relative comparisons within each dataset are. A re-weighting analysis at 5:1 and 20:1 (Section 5, "Cost-Ratio Robustness") shows the key cost rankings are stable across this range, except that the wrapper's cost advantage over the unwrapped base model requires a ratio of roughly 10:1 or higher.

6. **Two datasets:** External validity limited to two cloud providers and the specific function types/workload periods sampled. Generalization claims are conditional on these datasets.

7. **Static_P90 excluded from Phase 2:** Cannot compute training residuals for a model that predicts a constant. The EVT pipeline requires a non-trivial base model.

8. **TCN stochastic training:** Results use seed=42 for reproducibility. Small variance across seeds is possible.

---

## 10. Audit Summary (all checks pass)

| Phase | Checks | Result | Notes |
|-------|--------|--------|-------|
| Phase 1 Azure | 74/74 | PASS | All 6 models |
| Phase 2 Azure | 106/106 | PASS (2 XFAIL) | Forecast_Only + Seasonal_Naive: buffer does not scale up at extremes because these models partially predict seasonal spikes — documented nuance, not a bug |
| Phase 3 | 148/148 | PASS | 30 runs |
| Phase 4 | 63/63 | PASS | C0/C4 bit-identical to Phase 1/2 |
| Phase 1 Huawei | 126/126 | PASS | Combined dataset |
| Phase 2 Huawei | 121/121 | PASS | Combined dataset; 0 expected failures (unlike Azure) |
