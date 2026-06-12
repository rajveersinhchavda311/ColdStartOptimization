# Cold Start Optimization: Serverless Provisioning with EVT-CVaR Risk Awareness

A research pipeline for serverless cold-start minimization. The project processes real-world serverless workload traces, evaluates forecasting baselines, and applies a dynamic risk-aware provisioning layer based on Extreme Value Theory (EVT) and Conditional Value-at-Risk (CVaR).

## Project Structure

```
ColdStartOptimization/
│
├── data/                          # Datasets (raw + processed)
│   ├── raw/                       # Original traces
│   └── processed/
│       ├── azure/                 # Azure train/val/test splits (11,232/3,744/3,744)
│       └── huawei/                # Huawei splits per region + combined (25,920/8,640/8,640)
│           ├── combined/
│           ├── R1/ R2/ R3/ R4/ R5/
│
├── preprocessing/                 # Data pipeline
│   ├── preprocess_azure.py        # Aggregate invocations → per-minute demand
│   ├── preprocess_huawei.py       # Huawei multi-region traces processing
│   └── feature_engineering.py    # Lag feature extraction (lag_1..lag_10 + lag_1440)
│
├── models/                        # Forecasting models
│   ├── base.py                    # Abstract BaseModel interface
│   ├── reactive.py                # Reactive: predict = lag_1
│   ├── static_p90.py              # Static P90: predict = P90(train)
│   ├── forecast_only.py           # Forecast Only: predict = mean(lag_1..lag_10)
│   ├── seasonal_naive.py          # Seasonal Naive: predict = lag_1440
│   ├── linear_seasonal.py         # Linear Seasonal: OLS on [lag_1, lag_1440]
│   ├── tcn.py                     # TCN: dilated causal convolutions
│   └── risk_aware.py              # RiskAwareModel: EVT-CVaR wrapper
│
├── evaluation/                    # Evaluation framework
│   ├── simulator.py               # Timestep-by-timestep provisioning simulator
│   ├── metrics.py                 # SLA, cost, extreme SLA metrics
│   ├── extreme.py                 # Extreme event detection (P99 threshold)
│   └── evt.py                     # EVT/GPD fitting (POT method)
│
├── scripts/                       # Execution scripts (one per phase)
│   ├── run_baselines.py           # Phase 1: all 6 baselines (Azure)
│   ├── run_audit.py               # Phase 1: audit (74/74)
│   ├── generate_phase1_graphs.py  # Phase 1: visualizations
│   ├── run_phase2.py              # Phase 2: risk-aware models (Azure)
│   ├── run_phase2_audit.py        # Phase 2: audit (106/106 PASS, 2 XFAIL)
│   ├── generate_phase2_graphs.py  # Phase 2: visualizations
│   ├── run_phase3.py              # Phase 3: sensitivity analysis (30 configs)
│   ├── audit_phase3.py            # Phase 3: audit (148/148)
│   ├── generate_phase3_graphs.py  # Phase 3: visualizations
│   ├── run_phase4.py              # Phase 4: ablation C1/C2/C3 (C0/C4 from disk)
│   ├── audit_phase4.py            # Phase 4: audit (63/63)
│   ├── generate_phase4_graphs.py  # Phase 4: visualizations
│   ├── run_phase1_huawei.py       # Phase 5: Phase 1 baselines on Huawei
│   ├── audit_phase1_huawei.py     # Phase 5: Phase 1 audit (Huawei)
│   ├── run_phase2_huawei.py       # Phase 5: Phase 2 risk-aware on Huawei
│   ├── audit_phase2_huawei.py     # Phase 5: Phase 2 audit (Huawei)
│   ├── generate_phase5_graphs.py  # Phase 5: visualizations
│   ├── cache_tcn_residuals.py     # Caches real TCN training residuals (gate-verified) for figures
│   ├── cost_ratio_sensitivity.py  # Re-weights frozen results under 5:1/10:1/20:1 cost ratios
│   └── evt_bootstrap_ci.py        # Parametric bootstrap 95% CIs for GPD xi / CVaR_z
│
├── results/
│   ├── phase1/
│   │   ├── azure/                 # Phase 1 Azure metrics, diagnostics, audit
│   │   └── huawei/combined/       # Phase 5: Phase 1 baselines on Huawei
│   ├── phase2/
│   │   ├── azure/                 # Phase 2 Azure metrics, EVT params, audit
│   │   └── huawei/                # Phase 5: combined/ + R1..R5/ EVT params
│   ├── phase3/azure/              # Phase 3 sensitivity configs + audit
│   ├── phase4/azure/              # Phase 4 ablation conditions + audit
│   ├── phase5/                    # Cross-dataset ξ summary (evt_xi_summary.csv)
│   └── analysis/                  # Cost-ratio sensitivity + EVT bootstrap CIs
│
├── graphs/
│   ├── preprocessing/             # Time series, histograms (Azure + Huawei regions)
│   ├── phase1/azure/              # SLA, cost, extreme event analysis
│   ├── phase2/azure/              # Dynamic buffer, SLA/cost comparison
│   ├── phase3/azure/              # Sensitivity curves, interaction plots
│   ├── phase4/azure/              # Ablation heatmap, incremental contributions
│   └── phase5/                    # Cross-dataset comparison, tail heaviness
│
└── docs/                          # Documentation
    ├── paper_context.md           # Master paper-writing context (all numbers + figures)
    ├── repository_structure.md    # Full project structure guide
    ├── preprocessing/             # Feature engineering methodology
    ├── phase1/                    # Phase 1 architecture + audit
    ├── phase2/                    # Phase 2 EVT-CVaR architecture + audit
    ├── phase3/                    # Phase 3 sensitivity analysis
    ├── phase4/                    # Phase 4 ablation study
    └── phase5/                    # Phase 5 Huawei generalization study
```

## Overview

This project investigates whether risk-aware provisioning can reduce serverless cold starts without excessive overprovisioning. It proceeds in five phases:

**Phase 1** establishes forecasting baselines evaluated by a deterministic provisioning simulator. Six models span the complexity spectrum from a zero-parameter rule to a trained deep learning model. Best wrappable forecaster: TCN (request SLA = 0.9859, extreme SLA = 0.9436, cost 371.4M). Static_P90 attains a higher request SLA (0.9925) at similar cost, but has the worst extreme SLA (0.8650) and — predicting a constant — produces no forecast-error distribution to risk-wrap in Phase 2.

**Phase 2** wraps each Phase 1 model with a dynamic EVT-CVaR safety buffer. The buffer scales with local forecast volatility, shrinking during calm periods and expanding during uncertain ones. Cold starts fall 88–98%. Best result: RiskAware(TCN), request SLA = 0.9997, extreme SLA = 0.9944, cost 342M.

**Phase 3** tests robustness: 30 parameter configurations across α, W, and threshold. All 30 remain above request SLA = 0.99. α is the primary cost–SLA lever; W and threshold are near-inert.

**Phase 4** decomposes the Phase 2 buffer into its two components (EVT vs Gaussian multiplier; dynamic vs static σ) via a 2×2 factorial ablation. Key finding: adding any calibrated buffer is the dominant effect (+1.33–1.44 pp SLA); EVT provides tail protection (extreme SLA 0.987–0.993 → 0.998–0.9999); dynamic σ is a cost optimizer.

**Phase 5** applies the frozen Azure methodology to Huawei without any modification. Cold starts fall 95–99%. EVT consistently recommends a 1.07–2.95× larger buffer than Gaussian across all 7 datasets, confirming heavy-tail behavior generalizes across cloud providers.

**Evaluation metric:** Total provisioning cost = `cold_starts × 10 + idle_capacity × 1`. Cold starts are penalized 10× more than idle capacity.

## Datasets

**Azure Functions 2019:** Public trace, 14 days, 1-minute resolution. Total invocations/minute used as demand proxy. Mean ~614K invocations/min. Train/val/test: 11,232/3,744/3,744.

**Huawei Cloud (Jan 2025):** 5 independent function regions (R1–R5) plus combined aggregate. 31 days, 1-minute resolution. Train/val/test: 25,920/8,640/8,640. Combined mean ~261 invocations/min.

**Note on "concurrency":** The field labeled `concurrency` is the total platform-wide invocations per minute (sum across all functions). True concurrency requires duration data, which is unavailable. This is a standard simplification in the serverless scheduling literature.

**Features:** lag_1..lag_10 + lag_1440 (24-hour daily seasonal lag). Burn-in: first 1,440 rows dropped to populate lag_1440.

## Phase 1: Forecasting Baselines

| Model | Strategy | Training Required |
|-------|----------|-------------------|
| Reactive | `lag_1` — last minute's demand | No |
| Static P90 | `P90(train)` — fixed constant | No |
| Forecast Only | `mean(lag_1..lag_10)` — moving average | No |
| Seasonal Naive | `lag_1440` — same minute yesterday | No |
| Linear Seasonal | OLS on `[lag_1, lag_1440]` | Yes (OLS) |
| TCN | Causal dilated 1D convolutions | Yes (gradient descent, seed=42) |

**Phase 1 results (Azure, test set):**

| Model | Request SLA | Extreme SLA | Total Cost | Cold Starts |
|-------|------------|------------|-----------|-------------|
| Reactive | 0.9848 | 0.9491 | 407.8M | 37,072,085 |
| Static_P90 | 0.9925 | 0.8650 | 370.9M | 18,318,487 |
| Forecast_Only | 0.9823 | 0.8966 | 475.8M | 43,258,850 |
| Seasonal_Naive | 0.9502 | 0.9012 | 1,262.8M | 121,420,546 |
| Linear_Seasonal | 0.9815 | 0.9383 | 478.1M | 45,160,587 |
| **TCN** | **0.9859** | **0.9436** | **371.4M** | **34,377,188** |

Phase 1 audit: **74/74 PASS**.

## Phase 2: Risk-Aware EVT-CVaR Provisioning

Each Phase 1 model (except Static_P90) is wrapped with RiskAwareModel:

```
final_prediction[t] = base_prediction[t] + sigma_t × CVaR_z
```

Where:
- `sigma_t` = std of the last W=30 forecast residuals (rolling volatility estimate)
- `CVaR_z` = Conditional Value-at-Risk at α=0.99 fitted to standardized training residuals via GPD/POT (Peaks Over Threshold with Generalized Pareto Distribution)

**EVT anchor parameters (fixed throughout Phases 2–5):** α=0.99, W=30, threshold=P90.

**Phase 2 results (Azure, test set):**

| Model | Request SLA | Extreme SLA | Total Cost | Cold Start Reduction |
|-------|------------|------------|-----------|---------------------|
| RiskAware(Reactive) | 0.9996 | 0.9969 | 446.6M | −97.7% |
| RiskAware(Forecast_Only) | 0.9996 | 0.9931 | 489.0M | −97.5% |
| RiskAware(Seasonal_Naive) | 0.9943 | 0.9790 | 410.9M | −88.6% |
| RiskAware(Linear_Seasonal) | 0.9997 | 0.9951 | 365.6M | −98.2% |
| **RiskAware(TCN)** | **0.9997** | **0.9944** | **342.4M** | **−97.8%** |

Phase 2 audit: **106/106 PASS (2 XFAIL)** — two expected failures are documented and counted as passed; see `docs/phase2/verification.md`.

**EVT vs Gaussian gap (Azure):**

| Model | CVaR_z (EVT) | K_GAUSSIAN (α=0.99) | Ratio |
|-------|-------------|---------------------|-------|
| RiskAware(Reactive) | 4.160 | 2.665 | 1.56× |
| RiskAware(TCN) | 4.295 | 2.665 | 1.61× |

The Gaussian assumption at the same confidence level would under-buffer by 33–61%.

## Phase 3: Sensitivity Analysis

**Purpose:** Verify the method is robust to hyperparameter choice.

**Design:** 9 one-at-a-time configs (α: 0.95/0.975/0.99; W: 10/30/60; threshold: P85/P90/P95) plus 8 full-factorial boundary configs (2×2×2). Total: 30 unique runs on Reactive and TCN.

**Findings:**
- All 30 configs: request SLA ∈ [0.9978, 0.99995] — all above 0.99 baseline
- **α is the primary lever:** α=0.95 → cost 239M, SLA 0.9989 (TCN); α=0.99 → cost 342M, SLA 0.9997 (30% cost difference for 0.08pp SLA relaxation)
- **W is near-inert on cost, weakly positive on SLA** (W=60 weakly dominates W=30)
- **Threshold is nearly inert:** SLA = 0.9997 at P85, P90, and P95 for TCN

Phase 3 audit: **148/148 PASS**.

## Phase 4: Ablation Study

**Purpose:** Decompose the EVT-CVaR buffer into independent components to determine which elements drive the performance gains.

**2×2 factorial design:**

| Condition | σ type | Multiplier | Label |
|-----------|--------|------------|-------|
| C0 | none | none | Phase 1 baseline |
| C1 | Static (σ_train) | K_GAUSSIAN = 2.665 | Fixed Gaussian buffer |
| C2 | Dynamic (rolling W=30) | K_GAUSSIAN = 2.665 | Adaptive Gaussian |
| C3 | Static (σ_train) | EVT CVaR_z | Fixed EVT buffer |
| C4 | Dynamic (rolling W=30) | EVT CVaR_z | Full Phase 2 |

**Results (Request SLA × 2 models):**

| Condition | Reactive SLA | Reactive Cost | TCN SLA | TCN Cost |
|-----------|-------------|--------------|---------|---------|
| C0 No Buffer | 0.9848 | 408M | 0.9859 | 371M |
| C1 Static+Gaussian | 0.9992 | 324M | 0.9992 | 239M |
| C2 Dynamic+Gaussian | 0.9987 | 315M | 0.9988 | 236M |
| C3 Static+EVT | **0.9999** | 473M | **0.9999** | 358M |
| C4 Dynamic+EVT (Phase 2) | 0.9996 | 447M | 0.9997 | 342M |

**Key findings:**
1. **C0→C1 (+1.33–1.44pp) is the dominant effect** — adding any calibrated buffer closes ~94% of cold starts
2. **EVT vs Gaussian:** extreme SLA improves from 0.987–0.993 (C1/C2) to 0.998–0.9999 (C3/C4) — an order of magnitude closer to perfect
3. **Dynamic σ = cost efficiency:** C3→C4 saves 4–6% cost at only 0.02–0.03pp SLA decrease
4. **C3 (Static+EVT) is the max-SLA corner; C4 (Phase 2) is the cost-optimal choice**

Phase 4 audit: **63/63 PASS**.

## Phase 5: Huawei Generalization

**Purpose:** External validity — apply the frozen Azure methodology to Huawei without modification.

**Zero-modification principle:** α=0.99, W=30, threshold=P90, cost function, feature set, and all model architectures are unchanged.

**Phase 2 results (Huawei Combined, test set):**

| Model | Request SLA | Extreme SLA | Cold Start Reduction |
|-------|------------|------------|---------------------|
| RiskAware(Reactive) | 0.9973 | 0.9554 | −98.7% |
| RiskAware(Forecast_Only) | 0.9968 | 0.9466 | −97.9% |
| RiskAware(Seasonal_Naive) | 0.9959 | 0.9376 | −96.2% |
| RiskAware(Linear_Seasonal) | 0.9960 | 0.9365 | −96.5% |
| RiskAware(TCN) | 0.9959 | 0.9339 | −95.5% |

**EVT generalizes:** CVaR_z / K_GAUSSIAN > 1.0 in all 14 (model, dataset) combinations across 7 datasets. TCN ξ > Reactive ξ in all 7 datasets.

**Extreme SLA gap:** Huawei extreme SLA (0.93–0.96) is lower than Azure (0.98–0.997) because Huawei test-set spikes reach 5× training P99 (vs 1.10× on Azure) — out-of-distribution demand EVT cannot anticipate.

Phase 5 audits: **126/126 PASS (Phase 1), 121/121 PASS (Phase 2)**.

## Setup and Usage

```bash
pip install -r requirements.txt               # pinned environment (Python 3.11)
```

> **Reproducibility note:** All stored results were generated under the pinned versions in
> `requirements.txt`. The GPD shape/scale parameters depend on `scipy.stats.genpareto.fit`
> (MLE), so a different scipy version may yield slightly different ξ values; TCN training is
> seeded (seed=42, cudnn-deterministic) and reproduces exactly on the same torch build.

```bash
# --- Phase 1: Forecasting Baselines (Azure) ---
python preprocessing/preprocess_azure.py       # raw → full_series.csv
python scripts/regenerate_features.py          # add lag features (lag_1-10 + lag_1440)
python scripts/run_baselines.py                # evaluate 6 models
python scripts/run_audit.py                    # 74/74 audit checks
python scripts/generate_phase1_graphs.py

# --- Phase 2: Risk-Aware EVT-CVaR (Azure) ---
python scripts/run_phase2.py
python scripts/run_phase2_audit.py             # 106/106 PASS (2 XFAIL)
python scripts/generate_phase2_graphs.py

# --- Phase 3: Sensitivity Analysis ---
python scripts/run_phase3.py                   # 30 configs
python scripts/audit_phase3.py                 # 148/148
python scripts/generate_phase3_graphs.py

# --- Phase 4: Ablation Study ---
python scripts/run_phase4.py                   # C1, C2, C3 (C0/C4 loaded from disk)
python scripts/audit_phase4.py                 # 63/63
python scripts/generate_phase4_graphs.py

# --- Phase 5: Huawei Generalization ---
python preprocessing/preprocess_huawei.py      # raw → Huawei splits (all regions)
python scripts/run_phase1_huawei.py
python scripts/audit_phase1_huawei.py
python scripts/run_phase2_huawei.py
python scripts/audit_phase2_huawei.py
python scripts/generate_phase5_graphs.py
```

## Key Design Principles

1. **No data leakage**: All models receive only `lag_k` columns (historical); `concurrency` is never accessed in `predict()`.
2. **Chronological splits**: Train → Val → Test, strictly in time order. No shuffling.
3. **Train-derived thresholds**: Extreme event threshold (P99 of training demand) and EVT parameters are computed from training data only.
4. **Sequential inference**: Phase 2 processes each test timestep in order, reconstructing volatility from past residuals via lag_1 — no hindsight.
5. **Reproducibility**: All deterministic models are verified to produce identical results across runs. TCN uses a fixed random seed (seed=42).
6. **Zero-modification generalization**: Phase 5 Huawei runs use the exact same α, W, threshold, and cost model as Azure with no tuning.

## Documentation

| Document | Description |
|----------|-------------|
| `docs/paper_context.md` | **Master paper-writing context** — all numbers, figures, narrative arc, terminology |
| `docs/repository_structure.md` | Complete project structure guide |
| `docs/pre_paper_cleanup.md` | Changelog for all pre-paper cleanup changes and audit fixes (June 2026) |
| `docs/preprocessing/preprocessing_guide.md` | Data processing, feature engineering, split methodology |
| `docs/phase1/architecture.md` | Phase 1 model and evaluation architecture |
| `docs/phase1/verification.md` | Phase 1 audit results (74/74) |
| `docs/phase2/architecture.md` | Phase 2 EVT-CVaR architecture and math |
| `docs/phase2/verification.md` | Phase 2 audit results (106/106 PASS, 2 XFAIL documented) |
| `docs/phase3/sensitivity_analysis.md` | Phase 3 design, results, and robustness findings |
| `docs/phase4/ablation_study.md` | Phase 4 ablation design, results, and paper narrative |
| `docs/phase5/generalization_study.md` | Phase 5 Huawei results, EVT parameters, cross-dataset analysis |
