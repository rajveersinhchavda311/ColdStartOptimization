# Phase 5: Huawei Generalization Study

## Overview

Phase 5 applies the EVT-CVaR provisioning methodology — developed and frozen on the Azure dataset through Phases 1–4 — to an independent Huawei serverless workload dataset. The purpose is external validation: demonstrating that the methodology generalizes beyond the platform on which it was designed.

**Target sentence for the paper:** "The methodology was developed entirely on Azure and applied to Huawei without modification."

---

## Huawei Dataset Description

**Source:** Huawei Cloud serverless function traces (January 2025)  
**Resolution:** Minute-level (1-minute intervals)  
**Time span:** 2025-01-01 to 2025-01-31 (31 days, 44,640 total rows)  
**Regions:** 5 independent function traces (R1–R5) plus a combined aggregate

**Region structure:**
- `combined`: Aggregate concurrency = R1 + R2 + R3 + R4 + R5 (verified exact sum)
- `R1`, `R2`, `R3`, `R4`, `R5`: Individual function invocation traces

**Demand statistics (training set, from split_info.json):**

| Region   | Mean     | Std      | P90      | P99      |
|----------|----------|----------|----------|----------|
| Combined | *(populated at run time)* | | | |
| R1       | *(populated at run time)* | | | |
| R2       | *(populated at run time)* | | | |
| R3       | *(populated at run time)* | | | |
| R4       | *(populated at run time)* | | | |
| R5       | *(populated at run time)* | | | |

**Comparison to Azure:**
- Azure: 18,720 post-burn-in rows (13 days at minute resolution), 60/20/20 → 11,232 / 3,744 / 3,744
- Huawei: 43,200 post-burn-in rows (30 days), 60/20/20 → 25,920 / 8,640 / 8,640
- Huawei test set is 2.3× larger; demand scale differs (Azure is higher-traffic)
- Both datasets have the same 1-minute resolution and daily periodicity

---

## Preprocessing Decisions

### Why lag_1440 is appropriate

The existing Huawei splits were generated without `lag_1440`. This was a bug: the daily seasonal lag is a core feature of the methodology (required by Seasonal_Naive, Linear_Seasonal, and TCN). It was regenerated.

`lag_1440` = concurrency 1440 minutes ago = same clock minute the previous day. At 1-minute resolution, this is identical in interpretation to Azure. The Huawei data spans 31 days (January 2025), so every training row has a valid `lag_1440` after the burn-in.

### Burn-in logic

The first 1440 rows of each series are dropped before splitting. This is the minimum required for `lag_1440` to be non-null. The same burn-in size is used for Azure (due to the same lag).

### Split sizes

| | Azure | Huawei |
|---|---|---|
| Raw rows | ~16,000+ | 44,640 |
| Burn-in dropped | 1,440 | 1,440 |
| Post-burn-in | 18,720 | 43,200 |
| Train (60%) | 11,232 | 25,920 |
| Val (20%) | 3,744 | 8,640 |
| Test (20%) | 3,744 | 8,640 |

All splits are strictly chronological with no overlap.

---

## The Zero-Modification Principle

Phase 5 applies the following constants from Azure Phase 2 **without any change**:

| Parameter | Value | Source |
|-----------|-------|--------|
| α (CVaR confidence) | 0.99 | Azure Phase 2 |
| W (volatility window) | 30 | Azure Phase 2 |
| EVT threshold percentile | P90 | Azure Phase 2 |
| Cost function | `cold × 10 + idle × 1` | Phases 1–4 |
| Feature set | lag_1..lag_10 + lag_1440 | Pre-Phase-3 fix |
| Model architectures | Unchanged | Phases 1–4 |
| Extreme event threshold | P99(train concurrency) | Phase 1 |

The only "adaptations" are mathematically necessary consequences of loading a different dataset (different train/val/test files). No hyperparameters were tuned on Huawei.

---

## EVT Parameter Tables

### Table 1: Azure vs Huawei Combined — All 5 Models

*(Values populated from `results/phase2/huawei/combined/evt_parameters.json` and `results/phase2/azure/evt_parameters.json` at run time)*

| Model | Dataset | ξ (shape) | β (scale) | u (P90 thresh) | CVaR_z | CVaR_z/K_G |
|-------|---------|-----------|-----------|----------------|--------|------------|
| RiskAware(Reactive)       | Azure         | −0.0928 | 1.1121 | 1.0326 | 4.1605 | 1.56× |
| RiskAware(Reactive)       | Huawei Comb.  | *(run)* | | | | |
| RiskAware(Forecast_Only)  | Azure         | +0.0026 | 0.9377 | 1.1661 | 4.2782 | 1.61× |
| RiskAware(Forecast_Only)  | Huawei Comb.  | *(run)* | | | | |
| RiskAware(Seasonal_Naive) | Azure         | +0.1845 | 0.5200 | 1.0761 | 3.5435 | 1.33× |
| RiskAware(Seasonal_Naive) | Huawei Comb.  | *(run)* | | | | |
| RiskAware(Linear_Seasonal)| Azure         | +0.0192 | 0.8942 | 1.1018 | 4.1606 | 1.56× |
| RiskAware(Linear_Seasonal)| Huawei Comb.  | *(run)* | | | | |
| RiskAware(TCN)            | Azure         | +0.0222 | 0.9265 | 1.1082 | 4.2949 | 1.61× |
| RiskAware(TCN)            | Huawei Comb.  | *(run)* | | | | |

*K_GAUSSIAN = 2.6652 (Gaussian CVaR at α=0.99, scipy-derived)*

### Table 2: Regional EVT Consistency — ξ Values for R1–R5

*(From `results/phase2/huawei/{R1..R5}/evt_parameters.json`)*

| Model | Azure | H.Comb | R1 | R2 | R3 | R4 | R5 |
|-------|-------|--------|----|----|----|----|-----|
| Reactive | −0.093 | *(run)* | | | | | |
| TCN      | +0.022 | *(run)* | | | | | |

*See `results/phase5/evt_xi_summary.csv` for the full populated table.*

---

## Phase 1 Results — Huawei Combined

*(From `results/phase1/huawei/combined/metrics.json`)*

| Model | Cold Starts | Request SLA | Extreme SLA | Total Cost |
|-------|-------------|-------------|-------------|------------|
| Reactive        | *(run)* | | | |
| Static_P90      | *(run)* | | | |
| Forecast_Only   | *(run)* | | | |
| Seasonal_Naive  | *(run)* | | | |
| Linear_Seasonal | *(run)* | | | |
| TCN             | *(run)* | | | |

## Phase 2 Results — Huawei Combined

*(From `results/phase2/huawei/combined/metrics.json`)*

| Model | Cold Starts | Request SLA | Extreme SLA | Cold-Start Reduction |
|-------|-------------|-------------|-------------|----------------------|
| RiskAware(Reactive)        | *(run)* | | | |
| RiskAware(Forecast_Only)   | *(run)* | | | |
| RiskAware(Seasonal_Naive)  | *(run)* | | | |
| RiskAware(Linear_Seasonal) | *(run)* | | | |
| RiskAware(TCN)             | *(run)* | | | |

---

## Interpretation

### On emotional attachment to results

The scientific claim is not "Huawei workloads have heavy tails." The claim is "EVT correctly adapts to the tail structure present in the data."

These are different claims. The first requires ξ > 0. The second is true for any ξ.

### Outcome 1: ξ > 0 and CVaR_z/K_GAUSSIAN > 1.3× (heavy tails)

Both Azure and Huawei serverless workloads exhibit heavy tails. EVT correctly detects and accounts for this on both platforms. The heavy-tail property appears to be a general characteristic of cloud function invocation patterns, not an Azure artefact. The multiplier ratio consistently above 1 across all models and both datasets confirms that the EVT buffer is larger than a Gaussian approximation would prescribe — meaning cold-start protection would be systematically under-estimated without EVT.

**Paper framing:** "Both platforms exhibit heavier-than-Gaussian forecast error tails; EVT correctly identifies and accounts for this, providing a buffer that would be undersized under a Gaussian assumption."

### Outcome 2: ξ ≈ 0 (near-Gaussian)

Huawei residuals behave approximately Gaussian. This is still publishable: EVT correctly adapts to the tail shape present in the data and converges toward a Gaussian-like solution when tails are light. The method degrades gracefully.

If CVaR_z/K_GAUSSIAN is close to 1.0, Gaussian would have been sufficient for Huawei. If EVT still produces a ratio > 1.1, there is still tail structure EVT captures that a Gaussian model would miss.

**Paper framing:** "Huawei residuals are approximately Gaussian; EVT adapts accordingly, converging toward the Gaussian multiplier. The method degrades gracefully when heavy tails are absent."

### Outcome 3: ξ < 0 (bounded tail)

EVT fits a bounded Weibull-type tail. The method remains leakage-free and correct. The EVT advantage is smaller on Huawei, but the framework still works correctly.

**Paper framing:** "Huawei residuals have a bounded tail; EVT fits the correct parametric family. The cold-start protection is maintained; the EVT multiplier is closer to Gaussian."

### What to look for in the results

Regardless of ξ sign:
- Is the cold-start reduction large on Huawei (Phase 1 → Phase 2)?
- Does extreme_sla improve significantly?
- Is the ratio CVaR_z/K_GAUSSIAN consistent across regions R1–R5?
- Is the ξ sign consistent across models within each dataset?

A consistent ξ sign across models on the same dataset is evidence that the tail behavior reflects the underlying workload, not model-specific prediction artifacts.

---

## Audit Results

### Phase 1 Audit

See `results/phase1/huawei/combined/audit_results.json`.

Key checks: test set identity across all 6 models, burn-in correctly applied (lag_1440 non-null throughout), accounting identity (cold_starts = max(demand − provisioned, 0)), no NaN/Inf values.

Expected outcome: all checks pass.

### Phase 2 Audit

See `results/phase2/huawei/combined/audit_results.json`.

Key checks: same as Phase 1 plus — no future leakage (first W=30 timesteps use sigma_train), buffer is dynamic (CV > 0.01), EVT parameters present and sane (ξ ∈ [−1, 2]).

**Known expected failures (XFAIL) — may or may not recur on Huawei:**
The Azure Phase 2 audit has two known XFAIL checks for "buffer larger during extreme events" for `RiskAware(Forecast_Only)` and `RiskAware(Seasonal_Naive)`. Root cause on Azure: these models' base predictions partially track the seasonal component (mean of recent lags vs. lag_1440 directly), causing residuals during demand spikes to sometimes be *smaller* than average (the spike was partially predicted). As a result, the dynamic sigma_t is not systematically higher during extreme events.

If these XFAILs recur on Huawei, they are documented as expected behavior, not bugs. If they do not recur (i.e., the buffer IS larger during extremes for these models on Huawei), this is noted as a positive finding and documented.

---

## Limitations

1. **31-day window:** Huawei data covers a single month (January 2025). Seasonal patterns beyond weekly are not captured. Azure Phase 1–4 uses ~13 days, so Huawei actually offers more temporal coverage, but neither dataset has multi-month data.

2. **Single platform per source:** Azure and Huawei represent one dataset per platform. Cloud workloads vary significantly by function type, region, and traffic pattern. The five Huawei regions provide intra-platform diversity, but cross-provider diversity is limited to two data points.

3. **Regions are not independent deployments:** R1–R5 are concurrent traces from the same Huawei infrastructure period. They may share workload drivers (time-of-day patterns, correlated traffic sources). The combined series is their exact sum, not an independent sample.

4. **Cost model is experimental:** The 10:1 cold:idle ratio is an assumption, not derived from cloud pricing. Absolute cost numbers are not comparable across platforms. SLA metrics (request_sla, extreme_sla) are unit-free and directly comparable.

5. **TCN training is stochastic:** TCN uses seed=42 for reproducibility, but results may vary slightly across hardware/CUDA versions.

---

## Connection to Paper Narrative

The paper's contribution is the EVT-CVaR framework and the methodology, not the specific value of ξ. Phase 5 serves three narrative functions:

1. **External validity:** Results on an independently collected dataset from a different cloud provider show the method works beyond its training environment.

2. **Generalization of the EVT choice:** If ξ > 0 on Huawei (or consistently near 0), this supports the claim that serverless workloads generally have non-Gaussian forecast error tails, making EVT a principled choice over a simpler Gaussian quantile buffer.

3. **Framework robustness:** The zero-modification principle — α, W, threshold, cost function, and feature set all unchanged — demonstrates that the methodology does not require per-platform tuning to function correctly.

The regional EVT table (ξ across R1–R5 + combined + Azure) is the paper's primary cross-dataset generalization evidence. The pattern in the ratio column (whether consistently > 1, mixed, or near 1) is the finding. Report it honestly.

---

## Files Reference

| File | Description |
|------|-------------|
| `results/phase1/huawei/combined/metrics.json` | Phase 1 baseline metrics |
| `results/phase1/huawei/combined/audit_results.json` | Phase 1 audit |
| `results/phase2/huawei/combined/evt_parameters.json` | EVT parameters (ξ, β, CVaR_z, ratio) for all 5 models |
| `results/phase2/huawei/combined/metrics.json` | Phase 2 risk-aware metrics |
| `results/phase2/huawei/combined/audit_results.json` | Phase 2 audit |
| `results/phase2/huawei/{R1..R5}/evt_parameters.json` | Regional EVT parameters |
| `results/phase5/evt_xi_summary.csv` | Full cross-dataset ξ summary table |
| `graphs/phase5/tail_heaviness_comparison.png` | Empirical dist + GPD fit (Fig 1) |
| `graphs/phase5/evt_multiplier_comparison.png` | CVaR_z/K_G bar chart (Fig 2) |
| `graphs/phase5/cold_start_reduction.png` | Phase 1→2 cold-start reduction (Fig 3) |
| `graphs/phase5/cross_dataset_phase2_sla.png` | SLA comparison (Fig 4) |
| `graphs/phase5/regional_evt_heatmap.png` | ξ heatmap across regions (Fig 5) |
| `graphs/phase5/evt_xi_summary.png` | Summary table figure (Fig 6) |
