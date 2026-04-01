# Phase 1: Architecture & Problem Definition

**Date**: April 1, 2026  
**Status**: Complete and Verified  
**Dataset**: Azure Functions (14 days, minute granularity, 4,030 test samples)

---

## 1. Problem Definition

### 1.1 Serverless Autoscaling Challenge

**Problem**: Serverless computing platforms (AWS Lambda, Azure Functions, Google Cloud Functions) must dynamically provision containers to handle variable demand while minimizing cost.

**Key Constraints**:
- **Cold starts**: New containers incur latency cost (100-500ms)
- **Idle resources**: Over-provisioned containers waste money
- **Extreme events**: Demand spikes occur unpredictably
- **Real-time decisions**: Autoscaling decisions are irrevocable once made

### 1.2 Why Cold Starts Matter

**Definition**: A cold start occurs when the number of running containers falls short of demand:
$$\text{cold\_start}[t] = \max(0, \text{demand}[t] - \text{containers}[t])$$

**Impact**:
1. **Service Failure**: Requests are dropped or timeout
2. **SLA Violations**: Service-level agreement penalties
3. **Customer Churn**: Users experience degraded performance
4. **Revenue Loss**: Lost requests and mandatory refunds
5. **Reputation Damage**: Trust erosion with customers

**Cost Model**: We model cold starts as 10× more costly than idle resources.

### 1.3 Why Extreme Events Matter

**Observation**: Demand is highly non-stationary with rare, severe spikes.

**Example**: On 1% of timesteps (extreme events, p99+), demand is 2-3× higher than average.

**Challenge**: Models optimized for average performance often:
- Fail catastrophically on rare spikes
- Allocate insufficient capacity during peaks
- Miss SLA targets precisely when they matter most

**Research Contribution**: Phase 1 establishes that **standard forecasting alone cannot handle extremes**. This motivates Phase 2's risk-aware methods.

---

## 2. Evaluation Framework

### 2.1 Lag-Based Features

We use **10 historical lags** as the feature vector:

$$\mathbf{x}[t] = [\text{lag}\_1[t], \text{lag}\_2[t], \ldots, \text{lag}\_{10}[t]]$$

where:
$$\text{lag}\_k[t] = \text{concurrency}[t-k]$$

represents the **container concurrency (demand) from k timesteps ago**.

**Chronological Invariant**: All lags satisfy $k \geq 1$, ensuring **no future lookahead** (features are historical only).

### 2.2 Cost Function

The autoscaling objective is to **minimize total cost**:

$$\text{Cost}[t] = C_{\text{idle}} \cdot \text{idle}[t] + C_{\text{cold}} \cdot \text{cold}[t]$$

where:
- $C_{\text{idle}} = 1.0$ (cost per idle container-second)
- $C_{\text{cold}} = 10.0$ (penalty per unserved request)
- $\text{idle}[t] = \max(0, \text{containers}[t] - \text{demand}[t])$
- $\text{cold}[t] = \max(0, \text{demand}[t] - \text{containers}[t])$

**Justification for 10:1 Ratio**:
- **Idle**: Container costs are <$0.02/hour = ~0.0056¢/second
- **Cold**: A failed request triggers:
  - Immediate SLA penalty (refund): 5-10% of transaction value
  - Operational overhead (error logging, alerting): 1% of transaction value
  - Customer churn impact (lost lifetime value): 50-100% of transaction value
- **Ratio**: Cold starts cost 10-100× more than idle provisioning in practice

**Total Cost** over test set:
$$\text{TotalCost} = \sum_{t=1}^{T} \text{Cost}[t]$$

### 2.3 Service-Level Agreement (SLA)

**Overall SLA Compliance**:
$$\text{SLA}_{\text{overall}} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[\text{cold}[t] = 0]$$

Fraction of **timesteps with perfect demand satisfaction** (no cold starts).

**Extreme Event SLA**:
$$\text{SLA}_{\text{extreme}} = \frac{\sum_{t \in \text{extreme}} \mathbb{1}[\text{cold}[t] = 0]}{|\text{extreme}|}$$

where $\text{extreme} = \{t : \text{demand}[t] \geq P99(\text{demand})\}$.

**Critical Insight**: A model can have good overall SLA but catastrophic extreme SLA (failure exactly when it matters most).

---

## 3. Model Evaluation Protocol

### 3.1 Per-Timestep Simulation

For each timestep $t$ in the test set:

1. **Extract features**: $\mathbf{x}[t] = [\text{lag}\_1[t], \ldots, \text{lag}\_{10}[t]]$
2. **Predict containers**: $\hat{c}[t] = \text{model}(\mathbf{x}[t])$
3. **Compute metrics**:
   - Cold starts: $\text{cold}[t] = \max(0, \text{demand}[t] - \hat{c}[t])$
   - Idle: $\text{idle}[t] = \max(0, \hat{c}[t] - \text{demand}[t])$
   - Cost: $\text{cost}[t] = C_{\text{idle}} \cdot \text{idle}[t] + C_{\text{cold}} \cdot \text{cold}[t]$
4. **Log result**: Store per-timestep metrics for later analysis

### 3.2 Aggregation

After evaluation, aggregate metrics across all timesteps:
- **Total Cold**: $\sum_t \text{cold}[t]$ (total unmet requests)
- **Total Idle**: $\sum_t \text{idle}[t]$ (total over-provisioning)
- **Total Cost**: $\sum_t \text{cost}[t]$ (combined objective)
- **SLA Overall**: $\frac{\#\text{(no-cold-start timesteps)}}{T}$

### 3.3 Extreme Event Analysis

For timesteps in the top 1% by demand (p99+):
- Separately report SLA compliance
- Report average cold starts per extreme timestep
- Identify failure patterns and degradation

**Example**: Static (P90) has 86.4% overall SLA but **0% extreme SLA** (complete failure on spikes).

---

## 4. No Data Leakage Guarantee

**Critical Requirement**: Evaluation must use only **historical information at decision time**.

**Implementation**:
- Features use only past lags: $\text{lag}_k[t]$ with $k \geq 1$
- Training data does not contain test data
- Chronological split: no shuffling, train-val-test are contiguous time windows

**Verification**:
- Line check: All predictions use $\mathbb{x}[t]$ constructed from $\text{lag}_k[t]$ with $k \geq 1$
- Temporal check: Test data never seen during model training
- Range check: Test timestamps all after training timestamps

**Impact**: Prevents unrealistic results where models "remember" future values.

---

## 5. Baseline Models (Design Space)

### 5.1 Reactive Baseline

$$\hat{c}_{\text{reactive}}[t] = \text{lag}\_1[t]$$

**Logic**: Provision equal to **previous timestep's demand**.

**Representation**: Minimal intelligence, reacts to observed load.

**Trade-off**:
- Low idle (responsive)
- High cold starts (reactive lag)

**SLA**: 54.89% overall, 24.39% extreme

### 5.2 Static (P90) Baseline

$$\hat{c}_{\text{static}}[t] = P90(\text{train\_concurrency})$$

**Logic**: Provision at **90th percentile of training demand**.

**Representation**: Conservative capacity planning, handles "normal" variability.

**Trade-off**:
- High idle (over-provisioned for average)
- Low cold starts on normal load
- **Complete failure on spikes**

**SLA**: 86.40% overall, **0% extreme**

### 5.3 Forecast-Only Baseline

$$\hat{c}_{\text{forecast}}[t] = \text{mean}(\text{lag}\_1[t], \ldots, \text{lag}\_{10}[t])$$

**Logic**: Simple **averaging** of past 10 timesteps.

**Representation**: Neutral forecasting heuristic (no risk-awareness).

**Trade-off**:
- Balanced cold starts and idle
- Moderate adaptation to trends

**SLA**: 56.87% overall, 2.44% extreme

### 5.4 MLP Forecast Baseline

$$\hat{c}_{\text{MLP}}[t] = \text{NN}_{\text{trained}}(\text{lag}\_1[t], \ldots, \text{lag}\_{10}[t])$$

**Architecture**: Neural network (MLP: 10 → 32 → 16 → 1)

**Logic**: Trained on **training data only**, learns nonlinear patterns.

**Representation**: Strong ML baseline without risk-awareness.

**Training**:
- Input: 10 lag features
- Output: Predicted concurrency
- Loss: MSE
- No regularization, no hyperparameter tuning

**Trade-off**:
- Better than Forecast-Only (captures nonlinearity)
- Still fails on extremes (no explicit spike modeling)

**SLA**: 49.18% overall, 7.32% extreme

---

## 6. Summary: Problem Motivation

### Key Gaps in Existing Approaches

1. **Static provisioning**: Cost-optimal but totally fails on spikes
2. **Reactive provisioning**: Handles spikes but costs 8.6% more  
3. **Standard ML forecasting**: Modest improvement (+7.32% extreme SLA) but insufficient

### Why Phase 2 is Needed

**Observation**: No baseline achieves **both** low cost AND high extreme-event SLA.

- Best cost: Static (P90) with 0% extreme SLA
- Best extreme resilience: Reactive with cost +8.6%
- ML forecasting: Intermediate, sufficient for neither objective

**Phase 2 Hypothesis**: Risk-aware methods (Extreme Value Theory + CVaR) can simultaneously:
1. Keep cost competitive with Static (P90) by optimizing for mean behavior
2. Improve extreme-event SLA through explicit tail modeling
3. Achieve balanced performance across normal and spike regimes

---

## References & Related Work

**Serverless Autoscaling**:
- Copik et al., "Extending Cloud-Native Applications with Cost-aware Autoscaling" (EuroSys 2021)
- García-Valls et al., "The World of Edge Computing" (IEEE Software 2018)

**Extreme Value Theory**:
- de Haan & Ferreira, "Extreme Value Theory: An Introduction" (Springer 2006)
- Embrechts et al., "Modelling Extremal Events" (Springer 1997)

**Risk-Aware Optimization**:
- Rockafellar & Uryasev, "Conditional Value-at-Risk for general loss distributions" (JAI 2002)
- Boyd & Vandenberghe, "Convex Optimization" (Cambridge 2004)

**Azure Functions Dataset**:
- Shahrad et al., "Serverless Computing: One Step Closer to an Ideal Computing Model" (USENIX HotCloud 2016)
- Azure public traces: https://doi.org/10.1145/3361525.3361535
