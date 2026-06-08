# Preprocessing Guide

This document outlines the preprocessing pipeline for the Serverless Workload traces (Azure Functions and Huawei), including the feature engineering choices and their methodological justification.

---

## 1. Raw Data Processing

The pipeline processes large-scale serverless traces to extract a per-minute platform-wide demand metric.

### Azure Functions Dataset (2019)

`preprocess_azure.py` reads the Azure public trace (14 days, minute granularity) and aggregates all per-function invocation counts into a single total per minute:

```python
concurrency_ts = df.groupby("timestamp")["invocations"].sum().reset_index()
concurrency_ts.columns = ["timestamp", "concurrency"]
```

The output is `data/processed/azure/full_series.csv` with columns `timestamp` and `concurrency`.

### On the "Concurrency" Label

The field is labeled `concurrency` for brevity, but it is technically **total platform-wide invocations per minute** — not true concurrency in the computer science sense.

**True concurrency** (simultaneous in-flight executions) requires both invocation counts and execution durations. The Azure 2019 public trace provides only invocation counts; execution durations are not available.

**Why this is acceptable as a provisioning proxy:**  
The number of function instances needed scales approximately linearly with invocation volume under roughly steady execution time. For scheduling and provisioning decisions — which must be made before functions execute — the invocation count is the standard available signal. This simplification is consistent with other work using the same dataset (e.g., Shahrad et al., 2020).

**Known limitation:**  
If execution durations vary substantially across time or function types, the invocation count will be an imperfect proxy. Cross-function duration heterogeneity is unobservable in this dataset. This is documented as a known limitation of the study's threat-to-validity.

### Huawei Dataset

`preprocess_huawei.py` performs analogous aggregation for Huawei public cloud traces, supporting both combined and multi-region (R1-R5) datasets.

---

## 2. Feature Engineering

`scripts/regenerate_features.py` enriches the raw time series with autoregressive lag features:

### Short-Term Lags (lag_1 to lag_10)

For each timestep $t$, `lag_k[t] = concurrency[t - k]`. The 10-lag window covers the immediate temporal context — useful for capturing minute-level oscillations and momentum.

### Daily Seasonal Lag (lag_1440)

`lag_1440[t] = concurrency[t - 1440]` is the observation from the same minute 24 hours earlier.

**Rationale:** Azure Functions workloads exhibit strong daily periodicity (human-driven usage patterns). A model cannot capture seasonal variation without a seasonal lag — lag_10 alone misses the 24-hour cycle entirely.

**Precedent:** This is the "seasonal naive" lag, standard in the forecasting literature (Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., Ch. 7).

### Feature Matrix Schema

Each row in the output CSV has:
```
timestamp | concurrency | lag_1 | lag_2 | ... | lag_10 | lag_1440
```

The `concurrency` column is the **target** for training; `lag_k` columns are the **features**. Models must never access `concurrency` during prediction (it would be future information at inference time).

---

## 3. Train/Validation/Test Splits

Chronological splitting is strictly enforced. No shuffling is performed.

### Split Ratios

| Split | Fraction | Rows |
|-------|----------|------|
| Train | 60% | 11,232 |
| Val   | 20% | 3,744 |
| Test  | 20% | 3,744 |
| **Total** | | **18,720** |

### Burn-In Period

The maximum lag used is lag_1440 (1440 rows). The first 1440 timesteps — which lack valid lag_1440 values — are discarded after feature computation. This is why total rows (18,720) are less than the raw series length.

The resulting time ranges are:
- **Train:** 2019-01-02 00:00 → 2019-01-09 ~19:11
- **Val:** 2019-01-09 ~19:12 → 2019-01-12 ~09:35  
- **Test:** 2019-01-12 ~09:36 → 2019-01-14 23:59

### Leakage Prevention

- Splits are computed and saved before any model sees the data
- Extreme event threshold is derived from **training data only** (`P99(train)`)
- EVT parameters (Phase 2) are fitted on **training residuals only**
- No test-set information is used to tune any hyperparameter

### Output Location

All splits are written to `data/processed/azure/{train,val,test}.csv`. This is the authoritative location read by all model runners.

Note: `preprocessing/feature_engineering.py` writes to `preprocessing/dataset/processed/` (the old path from the original implementation). Use `scripts/regenerate_features.py` for the current feature set with lag_1440.

---

## 4. Demand Statistics (Training Set)

For reference, the training-set demand distribution:

| Statistic | Value |
|-----------|-------|
| Mean | ~613,900 invocations/min |
| Std dev | ~68,696 |
| P90 | ~696,264 |
| P99 (extreme threshold) | ~785,458 |
| Min | ~300K (approx.) |
| Max | ~900K (approx.) |

These statistics are used for:
- **P99** → extreme event threshold (frozen from training, applied to test)
- **Mean/Std** → TCN target normalization
