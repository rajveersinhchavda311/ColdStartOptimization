# Phase 1: Architecture Document

## System Overview

Phase 1 implements a baseline ecosystem and evaluation framework for serverless cold start optimization. The goal is to establish rigorous, reproducible baselines that can later support publication-quality comparison against risk-aware methods.

### Core Question
> **Is forecasting alone sufficient for serverless autoscaling, or do we need risk-aware provisioning?**

Phase 1 answers this by evaluating four progressively sophisticated forecasting strategies against actual demand, measuring cost and SLA metrics.

---

## Data Flow

```
Preprocessed Data (FROZEN)
    data/processed/azure/{train,val,test}.csv
         |
         v
    +-------------+
    |   Models     |  fit(train_df) -> predict(test_df)
    |  (4 baselines)|
    +-------------+
         |
         v
    +----------------+
    | Simulator      |  timestep-by-timestep provisioning
    | (evaluation/)  |  provisioned = ceil(prediction)
    +----------------+
         |
         v
    +----------------+
    | Metrics        |  cost, SLA, extreme SLA
    | (evaluation/)  |  with mathematical validation
    +----------------+
         |
         v
    +-----------+     +-----------+
    | Results   |     | Graphs    |
    | (results/ |     | (graphs/  |
    |  phase1/) |     |  phase1/) |
    +-----------+     +-----------+
```

---

## Component Architecture

### 1. Models (`models/`)

All models implement the `BaseModel` abstract class:

| Method | Purpose |
|--------|---------|
| `fit(train_df)` | Configure/train using training data only |
| `predict(df)` | Generate predictions using ONLY lag columns |
| `name` | Human-readable identifier |
| `description` | Strategy description |

**Contract**: `predict()` must NEVER access the `concurrency` column. Only `lag_1` through `lag_10` are permitted inputs. Violating this constitutes data leakage.

#### Model Inventory

| Model | Strategy | Parameters | Training Required |
|-------|----------|------------|-------------------|
| Reactive | `lag_1` | 0 | No |
| Static P90 | `P90(train)` | 1 (frozen) | No (stat extraction) |
| Forecast Only | `mean(lag_1..lag_10)` | 0 | No |
| TCN | Causal dilated convolutions | ~22K | Yes (gradient descent) |

### 2. Evaluation Framework (`evaluation/`)

#### `extreme.py` - Extreme Event Analysis
- Computes P99 threshold from TRAINING data exclusively
- Flags extreme events in any dataset using this frozen threshold
- **Leakage guard**: Function signature accepts only `train_df`, making it structurally impossible to pass test data

#### `metrics.py` - Metric Computation
- Computes all metrics from simulation results
- Contains built-in mathematical validation assertions
- Documents cost model as experimental assumption

#### `simulator.py` - Provisioning Simulator
- Timestep-by-timestep simulation
- Provisioning policy: `provision = ceil(prediction)`
- Built-in invariant validation

### 3. Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `run_baselines.py` | Run all models, save results |
| `generate_graphs.py` | Generate publication-quality graphs |
| `run_audit.py` | Automated validation audit |

---

## Design Decisions

### D1: Provisioning = Prediction
In Phase 1, the provisioned capacity at each timestep equals the model's prediction (rounded up). There is no separate provisioning policy layer. This is deliberate: we want to isolate the forecasting question before introducing risk-aware logic.

### D2: Cost Model
- `c_cold = 10` (penalty per cold-started request)
- `c_idle = 1` (waste per idle slot)
- This is an **experimental assumption**, not derived from cloud-provider pricing
- The 10:1 ratio reflects the general principle that user-facing failures (cold starts) are significantly more costly than wasted resources

### D3: P99 Extreme Threshold
- Extreme events are defined as demand exceeding P99 of the training set
- P99 was chosen (over P95) to focus on truly exceptional spikes
- The threshold is computed once from training data and never updated

### D4: Chronological Evaluation
- All evaluation is strictly chronological
- No shuffling, no cross-validation (inappropriate for time series)
- Train -> Val -> Test split is enforced at the preprocessing level

---

## TCN Architecture

The TCN uses dilated causal 1D convolutions, NOT dense/MLP layers:

```
Input (batch, 1, 10)
    |
    v
TemporalBlock 0: dilation=1, channels 1->32
    |
    v
TemporalBlock 1: dilation=2, channels 32->32
    |
    v
TemporalBlock 2: dilation=4, channels 32->32
    |
    v
TemporalBlock 3: dilation=8, channels 32->32
    |
    v
Take last timestep features[:, :, -1]
    |
    v
Linear(32, 1) -> prediction
```

Each TemporalBlock contains:
- 2 CausalConv1d layers (left-padded, weight-normalized)
- ReLU activations + dropout
- Residual connection (1x1 conv for channel mismatch)

**Receptive field**: 1 + 2*(3-1)*(1+2+4+8) = 61 timesteps (covers full 10-lag input)

**Causal guarantee**: Left-padding ensures output[t] depends only on input[<=t].

---

## Metrics Definitions

| Metric | Formula | Verified By |
|--------|---------|-------------|
| Request SLA | `1 - (sum(cold_starts) / sum(demand))` | Assertion in `metrics.py` |
| Extreme SLA | `1 - (sum(cold_starts[extreme]) / sum(demand[extreme]))` | Assertion in `metrics.py` |
| Total Cost | `sum(cold_starts)*10 + sum(idle)*1` | Assertion in `metrics.py` |

Where `extreme` events are timesteps where `demand > P99(train_concurrency)`.
