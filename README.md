# Cold Start Optimization: Serverless Provisioning with EVT-CVaR Risk Awareness

A research pipeline for serverless cold-start minimization. The project processes real-world serverless workload traces, evaluates forecasting baselines, and applies a dynamic risk-aware provisioning layer based on Extreme Value Theory (EVT) and Conditional Value-at-Risk (CVaR).

## Project Structure

```
ColdStartOptimization/
│
├── data/                          # Datasets (raw + processed)
│   ├── raw/                       # Original traces
│   └── processed/azure/           # Preprocessed time series + splits
│       ├── full_series.csv        # Aggregated time series
│       ├── train.csv              # 60% chronological split (11,232 rows)
│       ├── val.csv                # 20% validation split  (3,744 rows)
│       └── test.csv               # 20% test split        (3,744 rows)
│
├── preprocessing/                 # Data pipeline
│   ├── preprocess_azure.py        # Aggregate invocations -> per-minute demand
│   ├── preprocess_huawei.py       # Huawei traces processing
│   └── feature_engineering.py    # Original lag feature extraction (lag_1..lag_10)
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
│   ├── metrics.py                 # SLA, cost metrics
│   ├── extreme.py                 # Extreme event detection (P99 threshold)
│   └── evt.py                     # EVT/GPD fitting (POT method)
│
├── scripts/                       # Execution scripts
│   ├── regenerate_features.py     # Regenerate features with lag_1440
│   ├── run_baselines.py           # Run all Phase 1 models
│   ├── run_phase2.py              # Run Phase 2 risk-aware models
│   ├── run_audit.py               # Phase 1 validation audit
│   └── run_phase2_audit.py        # Phase 2 validation audit
│
├── results/
│   ├── phase1/azure/              # Phase 1 results, metrics, audit
│   └── phase2/azure/              # Phase 2 results, EVT parameters, audit
│
└── docs/                          # Documentation
    ├── preprocessing/             # Preprocessing and feature engineering
    ├── phase1/                    # Phase 1 architecture, verification
    ├── phase2/                    # Phase 2 architecture, verification
    └── pre_phase3_changes.md      # Change log for pre-Phase-3 fixes
```

## Overview

This project investigates whether risk-aware provisioning can reduce serverless cold starts without excessive overprovisioning. It proceeds in phases:

**Phase 1** establishes forecasting baselines evaluated by a deterministic provisioning simulator. Six models span the complexity spectrum from a zero-parameter rule to a trained deep learning model.

**Phase 2** wraps each Phase 1 model with a dynamic EVT-CVaR safety buffer. The buffer scales with local forecast volatility, shrinking during calm periods and expanding during uncertain periods.

**Evaluation metric:** Total provisioning cost = `cold_starts × 10 + idle_capacity × 1`. Cold starts are penalized 10× more than idle capacity (user-facing failures vs. wasted resources).

## Datasets

- **Azure Functions 2019**: Public trace, 14 days, minute granularity. Total invocations per minute used as a provisioning demand proxy. ~600K invocations/minute on average.
- **Huawei**: Huawei public cloud traces (2025), multi-region (R1-R5).

**Note on "concurrency":** The field labeled `concurrency` in our dataset is the total platform-wide invocations per minute (sum across all functions). True concurrency would require duration data, which is unavailable. This aggregate is used as a provisioning demand proxy — a standard simplification in the serverless scheduling literature.

## Phase 1: Forecasting Baselines

Six models ordered by complexity:

| Model | Strategy | Training Required |
|-------|----------|-------------------|
| Reactive | `lag_1` — last minute's demand | No |
| Static P90 | `P90(train)` — fixed constant | No |
| Forecast Only | `mean(lag_1..lag_10)` — moving average | No |
| Seasonal Naive | `lag_1440` — same minute yesterday | No |
| Linear Seasonal | OLS on `[lag_1, lag_1440]` | Yes (OLS) |
| TCN | Causal dilated 1D convolutions | Yes (gradient descent) |

**Phase 1 results (Azure, test set):**

| Model | Request SLA | Extreme SLA | Total Cost |
|-------|------------|------------|-----------|
| Reactive | 0.9848 | 0.9491 | 407.8M |
| Static_P90 | 0.9925 | 0.8650 | 370.9M |
| Forecast_Only | 0.9823 | 0.8966 | 475.8M |
| Seasonal_Naive | 0.9502 | 0.9012 | 1,262.8M |
| Linear_Seasonal | 0.9815 | 0.9383 | 478.1M |
| **TCN** | **0.9859** | **0.9436** | **371.4M** |

Phase 1 audit: **74/74 PASS**.

## Phase 2: Risk-Aware EVT-CVaR Provisioning

Each Phase 1 model (except Static_P90) is wrapped with RiskAwareModel:

```
final_prediction[t] = base_prediction[t] + sigma_t × CVaR_z
```

Where:
- `sigma_t` = std of the last 30 forecast residuals (rolling volatility)
- `CVaR_z` = Conditional Value-at-Risk at α=0.99 fitted to standardized training residuals via GPD/POT

**Phase 2 results (Azure, test set):**

| Model | Request SLA | Extreme SLA | Total Cost | Cold Start Reduction |
|-------|------------|------------|-----------|---------------------|
| RiskAware(Reactive) | 0.9996 | 0.9969 | 446.6M | -97.7% |
| RiskAware(Forecast_Only) | 0.9996 | 0.9931 | 489.0M | -97.5% |
| RiskAware(Seasonal_Naive) | 0.9943 | 0.9790 | 410.9M | -88.6% |
| RiskAware(Linear_Seasonal) | 0.9997 | 0.9951 | 365.6M | -98.2% |
| **RiskAware(TCN)** | **0.9997** | **0.9944** | **342.4M** | **-97.8%** |

Phase 2 audit: **104/106** (2 expected failures — see `docs/phase2/verification.md`).

## Setup and Usage

```bash
# 1. Process Azure traces (raw -> full_series.csv)
python preprocessing/preprocess_azure.py

# 2. Generate feature matrix with lag_1..lag_10 + lag_1440
python scripts/regenerate_features.py

# 3. Run Phase 1 baselines
python scripts/run_baselines.py

# 4. Run Phase 2 risk-aware models
python scripts/run_phase2.py

# 5. Validate Phase 1 (74/74 checks)
python scripts/run_audit.py

# 6. Validate Phase 2 (104/106 checks)
python scripts/run_phase2_audit.py
```

## Key Design Principles

1. **No data leakage**: All models receive only `lag_k` columns (historical); `concurrency` is never accessed in `predict()`.
2. **Chronological splits**: Train → Val → Test, strictly in time order. No shuffling.
3. **Train-derived thresholds**: Extreme event threshold (P99 of training demand) and EVT parameters are computed from training data only.
4. **Sequential inference**: Phase 2 processes each test timestep in order, reconstructing volatility from past predictions — no hindsight.
5. **Reproducibility**: All deterministic models are verified to produce identical results across runs. TCN uses a fixed random seed.

## Documentation

| Document | Description |
|----------|-------------|
| `docs/preprocessing/preprocessing_guide.md` | Data processing, feature engineering, split methodology |
| `docs/phase1/architecture.md` | Phase 1 model and evaluation architecture |
| `docs/phase1/verification.md` | Phase 1 audit results |
| `docs/phase2/architecture.md` | Phase 2 EVT-CVaR architecture and math |
| `docs/phase2/verification.md` | Phase 2 audit results and expected failure analysis |
| `docs/pre_phase3_changes.md` | Change log for all pre-Phase-3 fixes |
