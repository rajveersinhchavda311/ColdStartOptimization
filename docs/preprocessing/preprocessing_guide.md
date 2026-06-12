# Preprocessing Guide

This document outlines the preprocessing pipeline for the Serverless Workload traces (Azure Functions and Huawei), including the feature engineering choices and their methodological justification.

---

## 1. Raw Data Processing

The pipeline processes large-scale serverless traces to extract a per-minute platform-wide demand metric.

### 1.1 Azure Functions Dataset (2019)

`preprocessing/preprocess_azure.py` reads the Azure public trace (14 days, minute granularity) and aggregates all per-function invocation counts into a single total per minute:

```python
concurrency_ts = df.groupby("timestamp")["invocations"].sum().reset_index()
concurrency_ts.columns = ["timestamp", "concurrency"]
```

The output is `data/processed/azure/full_series.csv` with columns `timestamp` and `concurrency`.

### 1.2 On the "Concurrency" Label

The field is labeled `concurrency` for brevity, but it is technically **total platform-wide invocations per minute** — not true concurrency in the computer science sense.

**True concurrency** (simultaneous in-flight executions) requires both invocation counts and execution durations. The Azure 2019 public trace provides only invocation counts; execution durations are not available.

**Why this is acceptable as a provisioning proxy:**
The number of function instances needed scales approximately linearly with invocation volume under roughly steady execution time. For scheduling and provisioning decisions — which must be made before functions execute — the invocation count is the standard available signal. This simplification is consistent with other work using the same dataset (e.g., Shahrad et al., 2020).

The same definition applies to Huawei: `concurrency(t)` = COUNT of requests in the 60-second window starting at t.

**Known limitation:** If execution durations vary substantially across time or function types, the invocation count will be an imperfect proxy. This is documented as a known limitation.

### 1.3 Huawei Dataset (2025)

**Source:** Huawei Public Cloud function traces, January 2025. 5 independent function traces (R1–R5), each spanning 31 days.

**Two-script pipeline:**

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `preprocessing/preprocess_huawei.py` | `data/raw/huawei/{R1..R5}/day_00.csv..day_30.csv` | `data/processed/huawei/{combined,R1-R5}/full_series.csv` | Aggregate raw event logs → per-minute time series |
| `scripts/preprocess_huawei.py` | `data/processed/huawei/*/full_series.csv` | `data/processed/huawei/*/train.csv`, `val.csv`, `test.csv`, `split_info.json` | Add lag features + burn-in + splits |

**Raw data structure:**
Each region has 31 day files (`day_00.csv` to `day_30.csv`). Key columns:
- `time`: cumulative seconds from trace start (base date: `2025-01-01 00:00:00`). This is NOT relative to the day boundary — it is global and monotonically increasing across all 31 files.
- `requestID`: unique event identifier (used only to verify no-duplicate assumption)
- `day`: partition label only, not used for timestamps

**Timestamp construction:**
```python
BASE_DATE = pd.Timestamp("2025-01-01 00:00:00")
timestamp = BASE_DATE + pd.to_timedelta(df["time"], unit="s")
```

**Concurrency counting:** each event timestamp is floored to its 60-second boundary; events per 60s bin are counted. Bins with no events are filled with zero. This produces a gap-free 1-minute time series.

**Combined vs regional time series:**
- `combined`: concatenate ALL events from R1–R5, then count per-minute. Mathematically identical to `R1 + R2 + R3 + R4 + R5` at every timestamp.
- **Additive invariant:** `combined[t] == R1[t] + R2[t] + R3[t] + R4[t] + R5[t]` for all t. Verified by `cross_validate()` in the preprocessing script (exact integer equality).
- All 6 time series share an identical `timestamp` column (same shared time index).

**Raw output:** 44,640 rows per region (31 days × 1,440 minutes/day).

---

## 2. Feature Engineering

Applied identically to both Azure and Huawei.

### 2.1 Short-Term Lags (lag_1 to lag_10)

For each timestep $t$, `lag_k[t] = concurrency[t - k]`. The 10-lag window covers the immediate temporal context — useful for capturing minute-level oscillations and momentum.

### 2.2 Daily Seasonal Lag (lag_1440)

`lag_1440[t] = concurrency[t - 1440]` is the observation from the same minute 24 hours earlier.

**Rationale:** Both Azure and Huawei workloads exhibit strong daily periodicity (human-driven usage patterns). A 10-minute input window (lag_1..lag_10) is blind to this cycle. lag_1440 = concurrency[t-1440] is exactly the same minute of the prior day, capturing the dominant seasonal signal without requiring a 1440-step sequence model.

**Precedent:** This is the "seasonal naive" lag, standard in the forecasting literature (Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., Ch. 7).

### 2.3 Feature Matrix Schema

Each row in the output CSV has:
```
timestamp | concurrency | lag_1 | lag_2 | ... | lag_10 | lag_1440
```

The `concurrency` column is the **target** for training; `lag_k` columns are the **features**. Models must never access `concurrency` during prediction (it would be future information at inference time).

---

## 3. Train/Validation/Test Splits

Chronological splitting is strictly enforced. No shuffling is performed at any stage.

### 3.1 Azure Splits

| Split | Fraction | Rows |
|-------|----------|------|
| Train | 60% | 11,232 |
| Val | 20% | 3,744 |
| Test | 20% | 3,744 |
| **Total post-burn-in** | | **18,720** |

**Burn-in:** 1,440 rows dropped (rows where lag_1440 would be NaN).

**Date ranges:**
- Train: 2019-01-02 00:00 → 2019-01-09 ~19:11
- Val: 2019-01-09 ~19:12 → 2019-01-12 ~09:35
- Test: 2019-01-12 ~09:36 → 2019-01-14 23:59

**Script:** `scripts/regenerate_features.py` — reads `data/processed/azure/full_series.csv`, adds lag_1..lag_10 + lag_1440, drops first 1440 rows, splits 60/20/20, writes `data/processed/azure/{train,val,test}.csv`.

Note: `preprocessing/feature_engineering.py` is the original Azure feature engineering script (only lag_1..lag_10, no lag_1440). It writes to `preprocessing/dataset/processed/` (old path). Do not use for current experiments — use `scripts/regenerate_features.py`.

### 3.2 Huawei Splits

Same split ratio and burn-in logic applied to all 6 Huawei subdirectories (combined, R1–R5).

| Split | Fraction | Rows |
|-------|----------|------|
| Train | 60% | 25,920 |
| Val | 20% | 8,640 |
| Test | 20% | 8,640 |
| **Total post-burn-in** | | **43,200** |

**Burn-in:** 1,440 rows dropped (same reason as Azure: lag_1440 requires 1440 prior observations).
- Raw: 44,640 rows → 43,200 post-burn-in.

**Date ranges (all 6 subdirectories share identical timestamps):**
- Train: 2025-01-02 00:00 → 2025-01-19 23:59
- Val: 2025-01-20 00:00 → 2025-01-25 23:59
- Test: 2025-01-26 00:00 → 2025-01-31 23:59

**Script:** `scripts/preprocess_huawei.py` — reads each region's `full_series.csv`, adds all lag features, applies burn-in, splits 60/20/20, writes `{train,val,test}.csv` and `split_info.json` to each region directory.

**split_info.json** (written per region) contains: row counts, timestamp boundaries, training-set demand statistics, column list, and a `"note"` field confirming the version. Example for combined:
```json
{
  "n_train": 25920,  "n_val": 8640,  "n_test": 8640,
  "train_start": "2025-01-02 00:00:00",  "test_end": "2025-01-31 23:59:00",
  "burn_in_rows_dropped": 1440,
  "note": "Splits regenerated for Phase 5 with lag_1440 added"
}
```

### 3.3 Leakage Prevention

- Splits are computed and saved before any model sees the data
- Extreme event threshold is derived from **training data only** (`P99(train)`)
- EVT parameters (Phase 2) are fitted on **training residuals only**
- No test-set information is used to tune any hyperparameter

---

## 4. Demand Statistics (Training Set)

### Azure

| Statistic | Value |
|-----------|-------|
| Mean | ~613,900 invocations/min |
| Std dev | ~68,696 |
| P90 | ~696,264 |
| P99 (extreme threshold) | ~785,458 |
| Max (train) | 1,258,768 |
| Test set max | 866,343 (~1.10× P99) |

### Huawei (all values are invocations/min from training set)

| Region | Mean | Std | P90 | P99 (extreme threshold) | Max (train) | Test set max |
|--------|------|-----|-----|------------------------|-------------|-------------|
| Combined | 260.7 | 130.3 | 429 | 729 | 1,902 | 3,657 (~5.0× P99) |
| R1 | 157.5 | 106.8 | 314 | 500 | 1,795 | — |
| R2 | 45.9 | 20.6 | 65 | 123 | 330 | — |
| R3 | 7.1 | 5.3 | 14 | 21 | 39 | — |
| R4 | 32.1 | 28.9 | 46 | 164 | 420 | — |
| R5 | 18.2 | 7.2 | 28 | 38 | 47 | — |

**Critical difference from Azure:** The Huawei combined test set contains demand spikes up to 3,657 — which is 5.0× the training P99 (729). Azure's test maximum is only 1.10× its training P99. This out-of-distribution spike magnitude explains why Huawei extreme SLA (0.93–0.96) is lower than Azure (0.98–0.997) despite similar cold-start reductions. See `docs/phase5/generalization_study.md` for analysis.

**Regional scale variation:** R3 is a tiny function (max=39 invocations/min); R1 dominates (mean=157.5). R4 and R2 show notably high P99/P90 ratios, indicating heavier tails and confirmed by their high GPD shape parameters ξ for TCN (R2: +0.50, R4: +0.54; Reactive ξ is +0.41 on R2 but −0.17 on R4).

These statistics are used for:
- **P99** → extreme event threshold (frozen from training, applied to test)
- **Mean/Std** → TCN target normalization
- **P90** → EVT threshold (Phase 2 anchor)
