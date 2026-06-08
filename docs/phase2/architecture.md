# Phase 2 Architecture: Dynamic EVT + CVaR Risk-Aware Provisioning

## Overview

Phase 2 introduces a risk-aware provisioning layer that wraps Phase 1
forecasting models. Instead of provisioning exactly the forecast
(`provision = prediction`, as in Phase 1), the risk layer adds a dynamic
safety buffer derived from Extreme Value Theory (EVT) and Conditional
Value-at-Risk (CVaR).

### Core Question

> **Can risk-aware provisioning reduce cold starts (improve SLA) without
> excessive overprovisioning, compared to forecast-only strategies?**

Phase 2 answers this by wrapping five Phase 1 forecasters with a
volatility-scaled EVT buffer and evaluating under identical conditions.

### Wrapped Models

| Base Model | Rationale for inclusion |
|-----------|------------------------|
| Reactive | Minimal baseline; lag-1 errors correlate well with demand volatility |
| Forecast_Only | Moving average; tests smoothing-model interaction with EVT buffer |
| Seasonal_Naive | Seasonal baseline; tests buffer on high-bias, low-variance base model |
| Linear_Seasonal | Interpretable learned model; direct linear competitor to TCN |
| TCN | Deep learning model; tests whether EVT further amplifies the best forecaster |

Static_P90 is excluded: constant predictions produce residuals equal to (demand − constant), which reflect demand variance rather than forecast error. Applying a volatility-scaled EVT buffer to an already-conservative static provisioner is semantically incoherent.

---

## Architecture

```
Phase 1 Model (frozen)
    |
    v
RiskAwareModel Wrapper
    |-- fit():
    |     1. Train base model on train_df
    |     2. Compute training residuals (actual - predicted)
    |     3. Compute sigma_train = std(residuals)
    |     4. Standardize: z = residuals / sigma_train
    |     5. Fit EVT (POT + GPD) on z -> obtain CVaR_z
    |
    |-- predict():
    |     For each timestep t (sequentially):
    |       1. base_pred[t] = base_model.predict(row_t)
    |       2. Reconstruct past residual from lag_1
    |       3. Update rolling volatility sigma_t
    |       4. buffer[t] = sigma_t * CVaR_z
    |       5. final[t] = base_pred[t] + buffer[t]
    |
    v
Existing Simulator (FROZEN)
    |-- provisioned = ceil(final_prediction)
    |-- cold_starts = max(0, demand - provisioned)
    |-- idle = max(0, provisioned - demand)
    |
    v
Existing Metrics (FROZEN)
    |-- Request SLA, Extreme SLA, Cost
```

---

## Mathematical Formulation

### Notation

| Symbol | Meaning |
|--------|---------|
| y_t | Actual demand at timestep t |
| y_hat_t | Base model forecast at timestep t |
| epsilon_t | Forecast residual: y_t - y_hat_t |
| sigma_t | Local volatility of residuals at t |
| z_t | Standardized residual: epsilon_t / sigma_t |
| u | POT threshold (P90 of z) |
| xi | GPD shape parameter |
| beta | GPD scale parameter |
| alpha | CVaR confidence level (0.99) |
| CVaR_z | CVaR of standardized distribution |
| B_t | Dynamic buffer at timestep t |
| P_t | Final provisioning target at t |
| W | Volatility window size (30) |

### Training Phase

1. **Fit base model**: Train the Phase 1 model on training data.

2. **Compute residuals**:
   epsilon_i = y_i - y_hat_i, for all i in training set

3. **Global volatility**:
   sigma_train = std(epsilon)

4. **Standardize residuals**:
   z_i = epsilon_i / sigma_train

5. **POT threshold selection**:
   u = P90(z)  (90th percentile of standardized residuals)

6. **Extract exceedances**:
   x_i = z_i - u, for all z_i > u

7. **Fit GPD via MLE**:
   x ~ GPD(xi, beta)
   Parameters (xi, beta) estimated via Maximum Likelihood.

8. **Value-at-Risk**:
   VaR(alpha) = u + (beta / xi) * [((1 - alpha) / P(z > u))^(-xi) - 1]

9. **Conditional Value-at-Risk**:
   CVaR(alpha) = VaR(alpha) / (1 - xi) + (beta - xi * u) / (1 - xi)
   Requires: xi < 1 for finite CVaR.

### Inference Phase (Sequential)

At each timestep t during evaluation:

1. **Base prediction**: y_hat_t = base_model.predict(row_t) [uses only lag columns]

2. **Reconstruct past residual** (for t > 0):
   epsilon_{t-1} = lag_1[t] - y_hat_{t-1}
   
   This uses lag_1 (which equals concurrency[t-1]) and our own past
   prediction. No future information is accessed.

3. **Update rolling volatility**:
   If enough history exists (>= W residuals):
     sigma_t = std(epsilon_{t-W}, ..., epsilon_{t-1})
   Otherwise (warm-up period):
     sigma_t = sigma_train

4. **Compute dynamic buffer**:
   B_t = sigma_t * CVaR_z

5. **Final prediction**:
   P_t = y_hat_t + B_t

### Why the Buffer is Dynamic

B_t = sigma_t * CVaR_z

- CVaR_z is a constant (fitted once on training data)
- sigma_t varies over time based on recent forecast errors
- During calm periods: small errors -> small sigma_t -> small buffer
- During volatile periods: large errors -> large sigma_t -> large buffer

This is fundamentally different from a static margin because the
provisioning aggressiveness adapts to the current uncertainty regime.

---

## EVT Methodology

### Target: Why Conditional Residuals?

We fit EVT on standardized residuals z_t rather than:
- **Raw demand** (non-stationary, violates EVT i.i.d. assumptions)
- **Raw residuals** (exhibit heteroskedasticity / volatility clustering)

Standardizing by sigma produces residuals closer to i.i.d., making
the GPD fit theoretically sound.

### Peak-Over-Threshold (POT)

Rather than fitting a distribution to all data (wasteful for tail
modeling), POT focuses only on extreme values above a threshold u.
The Pickands-Balkema-de Haan theorem guarantees that exceedances
over a sufficiently high threshold follow a GPD.

### Generalized Pareto Distribution (GPD)

The GPD has CDF:
  F(x) = 1 - (1 + xi * x / beta)^(-1/xi)

- xi > 0: heavy tail (Pareto-like), common in IT workloads
- xi = 0: exponential tail
- xi < 0: bounded tail

---

## Leakage Analysis

### What information is available at each timestep t?

| Information | Available? | Source |
|-------------|-----------|--------|
| lag_1[t] = concurrency[t-1] | Yes | DataFrame column (past) |
| lag_2[t] ... lag_10[t] | Yes | DataFrame columns (past) |
| concurrency[t] | NO | Target at current timestep |
| concurrency[t+1] | NO | Future |
| base_preds[0..t-1] | Yes | Our own past predictions |
| base_preds[t] | Yes | Generated from lag columns only |
| residuals[0..t-1] | Yes | Reconstructed from lag_1 - base_pred |

### Sigma initialization

The first W timesteps lack sufficient residual history. We use
sigma_train (computed entirely from training data) as the initial
volatility estimate. This is conservative and leakage-free.

---

## Design Decisions

1. **Wrapper pattern**: RiskAwareModel wraps Phase 1 models without
   modifying them. The base model is completely unaware of the risk layer.

2. **EVT on standardized residuals**: Theoretically justified for
   heteroskedastic time series. Makes GPD fit more robust.

3. **Rolling window volatility**: Simpler than GARCH, more robust
   on noisy IT telemetry, and computationally trivial.

4. **Static_P90 excluded**: Constant predictions produce zero-variance
   residuals. Volatility-scaled EVT is meaningless for constant forecasters.

5. **Sequential predict()**: Volatility is updated one step at a time
   to avoid hindsight bias. This is critical for research validity.

---

## Assumptions and Limitations

1. **Window size W=30 is fixed**: Not tuned. A sensitivity analysis
   could explore W in {10, 30, 60} but is deferred to a later phase.

2. **Single alpha=0.99**: No sweep over confidence levels yet.

3. **EVT threshold P90**: Fixed at 90th percentile. Alternative
   threshold selection methods (e.g., mean excess plot) not explored.

4. **Stationarity assumption**: The GPD is fit once on training data
   and applied globally. If the residual distribution shifts
   significantly in the test set, the fitted parameters may be stale.

5. **Cost model unchanged**: Same c_cold=10, c_idle=1 as Phase 1.
   The buffer will necessarily increase idle cost to reduce cold cost.

---

## Reproducibility

- All Phase 1 seeds remain unchanged.
- EVT fitting uses scipy.stats.genpareto (deterministic MLE).
- Rolling volatility is deterministic given the same input sequence.
- Results are fully reproducible for all deterministic base models
  (Reactive, Forecast_Only, Seasonal_Naive, Linear_Seasonal) — verified
  by Audit 10 (two independent runs produce bit-identical predictions).
- TCN reproducibility depends on the underlying PyTorch seed behavior
  (verified in Phase 1 audit).

## Assumptions and Limitations (Updated)

See the original assumptions list above. Additional limitations surfaced
by the 5-model comparison:

6. **Smoothing-based models exhibit inverted buffer behavior during extremes.**
   Forecast_Only and Seasonal_Naive produce lower local volatility (sigma_t)
   during demand ramp-ups than during normal oscillation. The EVT buffer still
   reduces cold starts by 88–97%, but the buffer does not systematically grow
   during extreme events for these models. See `docs/phase2/verification.md`
   for full analysis.

7. **Seasonal_Naive is a weak base for risk wrapping.** With a very high
   Phase 1 cold-start rate (121M), the EVT buffer can only partially compensate.
   RiskAware(Seasonal_Naive) remains the weakest Phase 2 model (SLA 0.9943 vs
   0.9996–0.9997 for others).
