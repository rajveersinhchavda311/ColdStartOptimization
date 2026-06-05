# Phase 1: Implementation Details

## Baseline Implementations

### 1. Reactive Baseline (`models/reactive.py`)

**Strategy**: `prediction[t] = lag_1[t] = concurrency[t-1]`

The simplest possible approach: provision whatever was needed one minute ago.

- **Input**: `lag_1` column only
- **Parameters**: 0
- **Training**: None
- **Expected behavior**:
  - Tracks steady-state well (most adjacent minutes are similar)
  - Lags behind during spikes (cold starts during rising demand)
  - Overprovisions during drops (idle capacity during falling demand)

### 2. Static P90 Baseline (`models/static_p90.py`)

**Strategy**: `prediction[t] = P90(train_concurrency)` for all t

A fixed provision level computed once from training data.

- **Input**: None during prediction (constant output)
- **Parameters**: 1 (the P90 value, frozen from training)
- **Training**: Extract P90 statistic from training concurrency
- **Leakage prevention**: P90 is computed ONLY from `train_df["concurrency"]`, never updated
- **Expected behavior**:
  - Very few cold starts (demand rarely exceeds P90)
  - High idle cost (most minutes have demand well below P90)
  - Represents the "overprovision constantly" strategy

### 3. Forecast-Only Baseline (`models/forecast_only.py`)

**Strategy**: `prediction[t] = mean(lag_1[t], ..., lag_10[t])`

A 10-minute simple moving average.

- **Input**: All 10 lag columns
- **Parameters**: 0
- **Training**: None
- **Expected behavior**:
  - Smoother than Reactive (averages out noise)
  - More responsive than Static P90 (follows demand trends)
  - Lags behind rapid changes more than Reactive (averaging introduces delay)

### 4. TCN Baseline (`models/tcn.py`)

**Strategy**: Genuine causal Temporal Convolutional Network

#### Architecture Details

```
Input: (batch, 1, 10) -- 10 lag values, oldest to newest
```

| Layer | Type | In Channels | Out Channels | Dilation | Kernel | Padding (left) |
|-------|------|-------------|--------------|----------|--------|---------------|
| Block 0 | TemporalBlock | 1 | 32 | 1 | 3 | 2 |
| Block 1 | TemporalBlock | 32 | 32 | 2 | 3 | 4 |
| Block 2 | TemporalBlock | 32 | 32 | 4 | 3 | 8 |
| Block 3 | TemporalBlock | 32 | 32 | 8 | 3 | 16 |
| Output | Linear | 32 | 1 | - | - | - |

Each TemporalBlock contains 2 CausalConv1d layers, so there are 8 total causal convolutions.

**Total parameters**: ~22,209

#### Input Ordering

Lag features are arranged chronologically:
- Position 0: `lag_10` (oldest, 10 minutes ago)
- Position 9: `lag_1` (newest, 1 minute ago)

This preserves the temporal structure that causal convolutions exploit.

#### Normalization

Z-score normalization using training data statistics:
- `normalized = (value - train_mean) / train_std`
- `train_mean` and `train_std` are computed from training lag values ONLY
- These statistics are frozen and applied identically to val/test

#### Training Protocol

- **Loss**: MSE (mean squared error)
- **Optimizer**: Adam, initial lr=1e-3
- **Scheduler**: ReduceLROnPlateau (patience=10, factor=0.5)
- **Early stopping**: patience=15 on validation MSE
- **Batch size**: 256
- **Max epochs**: 200
- **Seed**: 42 (PyTorch, NumPy, Python random)

#### Causality Verification

The `verify_causality()` method performs three checks:
1. **Padding verification**: Each CausalConv1d has left-padding = (kernel-1)*dilation
2. **Perturbation test 1**: Changing the last input position changes the output (it's visible)
3. **Perturbation test 2**: Changing the first position also changes the output (within receptive field)
4. **Receptive field**: RF = 61 >= 10 (covers full input)

---

## Evaluation Framework

### Simulator (`evaluation/simulator.py`)

For each timestep t:
1. Model generates prediction `p[t]` from lag features
2. Actual demand `d[t]` = concurrency column
3. Provisioned = `ceil(p[t])` (integer slots, minimum 0)
4. Cold starts = `max(0, d[t] - provisioned[t])`
5. Idle capacity = `max(0, provisioned[t] - d[t])`
6. Cost = `cold_starts * 10 + idle * 1`

**Invariants validated**:
- No negative values
- Cold starts and idle never coexist at same timestep
- Accounting identity: `cold = max(0, demand - provisioned)` exactly
- No NaN/Inf values

### Metrics (`evaluation/metrics.py`)

**Request SLA** = 1 - (total_cold_starts / total_demand)

This measures the fraction of ALL requests that were successfully served.
A Request SLA of 0.98 means 98% of requests were served without cold starts.

**Extreme SLA** = 1 - (cold_extreme / demand_extreme)

where extreme events are timesteps where demand > P99(train).
This measures performance specifically during extreme demand spikes.

Both formulas are verified with `assert abs(computed - expected) < 1e-10` after computation.

### Extreme Event Analysis (`evaluation/extreme.py`)

**Threshold**: P99 of training concurrency
- Computed ONCE from `train_df["concurrency"]`
- NEVER recomputed on val/test data
- NEVER derived from results/outcomes
- Function signature structurally prevents test data access

---

## Cost Model

| Parameter | Value | Justification |
|-----------|-------|---------------|
| c_cold | 10 | Cold starts cause latency, user-facing failures |
| c_idle | 1 | Idle resources waste money but don't harm users |
| Ratio | 10:1 | Standard in autoscaling literature |

**IMPORTANT**: This is an experimental assumption, not derived from any specific cloud provider's pricing schedule.

---

## Reproducibility

### Fixed Seeds
- Python random: seed=42
- NumPy: seed=42
- PyTorch: seed=42
- CUDA (if available): seed=42

### Deterministic Settings
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- DataLoader uses seeded generator

### File-Based Results
All results are saved to `results/phase1/azure/`:
- Per-model CSV results
- Consolidated comparison CSV
- Full metrics JSON
- Human-readable summary
