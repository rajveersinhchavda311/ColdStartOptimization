# Cold Start Optimization: Master Documentation

## 1. System Goal
The core objective of this research is to develop, validate, and benchmark a novel serverless autoscaling architecture. The primary constraint of modern Serverless computing (function-as-a-service) architectures is the "Cold Start penalty"—the delay incurred when instantiating new execution containers in response to unforeseen demand. Because this penalty is an order of magnitude higher than the cost of maintaining idle containers ($C_{cold} = 10 \times C_{idle}$), a superior autoscaler must excel in mitigating cold starts during extreme P99 demand spikes. 

---

## 2. Dataset & Preprocessing (Phase 1)
We evaluated our system against two massive real-world serverless traces:
-   **Azure Functions Dataset**: Contains ~16,000 distinct minute-by-minute invocation traces covering variable load patterns over multiple weeks.
-   **Pre-Processing**: The data was structurally cleaned, timestamped, combined, and extended with an autoregressive feature set (10 historical lag variables) mapping $t_{-1}$ to $t_{-10}$ onto target $t$. 
-   **Train/Test Split**: Enforced rigidly chronologically (75% Train/25% Test) to categorically avoid forward-looking data leakage.

---

## 3. Evaluation Pipeline & Baselines
The system processes timesteps sequentially, evaluating decision policies $c^*$ against the rolling ground truth. We defined three control baselines:
1.  **Reactive Scaler ($c_t = lag_1$)**: Always provisions to match the previous minute exactly.
2.  **Static Scaler ($P_{90}$)**: Provisions a constant capacity cap equal to the historical 90th percentile demand. Guaranteed massive idle waste but highly robust.
3.  **MLP Forecast Scaler**: A multi-layer neural network evaluating the lag state to predict the precise point-demand. Fails systemically during extreme demand spikes due to minimizing structural MSE.

---

## 4. Phase 2: Risk-Aware Architecture (EVT + CVaR)
Our principal contribution addresses the systemic structural failures of point forecasters. 

### Step 1: Quantile Bound Evaluation
A Gradient Boosting Regressor model predicts the probabilistic demand limits ($P_{50}, P_{90}, P_{99}$) locally for the next timestep using the pinball loss function. 

### Step 2: The "Risk Gate" & Extreme Value Theory
We extract the historical threshold $u=P_{99_{train}}$. If the local prediction $Q_{99} \ge u$, the system enters "Extreme Mode" and triggers the injection of Peaks-Over-Threshold scenarios derived from a Generalized Pareto Distribution (GPD). Otherwise, the system draws scenarios purely from the standard quantile distribution.

### Step 3: CVaR Optimization
Given the 300 generated scenarios $S$, the model computes a Conditional Value-at-Risk objective targeting the top 1% (worst-case) scenarios. 
$$ c^* = \text{argmin}_C \; \mathbb{E}[Cost(C, S) | Cost(C, S) \ge P_{99}(Costs)] $$
This strictly bounds the container candidate sweeps to precisely minimal $c^*$ outputs without inflating idle waste unnecessarily.

---

## 5. Final Results (Azure Data)

The "Conditionally Risk-Aware" V2 model achieved the stated objectives comprehensively:

-   **Extreme Reliability**: The model retained **70.73% SLA compliance** during the top 1% of raw demand spikes, obliterating the pure MLP Forecaster (7.3%). 
-   **Tractable Efficiency**: Instead of blindly adding a 50%+ fixed buffer, the V2 logic held a provisioning ratio of strictly **1.08x**, dropping raw cold start rates (43M to 4.1M), lowering aggregate total operational costs from $462.2M (MLP) to **$269.1M** (Risk-Aware).
-   **Precision Gating**: The model selectively utilized the computationally-expensive EVT path safely on **8.26%** of timesteps without defaulting to pure paranoia.

---

## 6. Limitations
-   The current pipeline executes locally and relies entirely on statistical lag extrapolation. This limits its ability to respond to external non-time-series phenomena (e.g., exogenous API rate limits upstream, or time-of-day seasonalities). 
-   The computational complexity of computing generating the GPD scenarios introduces latency strictly bound to $O(N \cdot M)$ CVaR optimizations per step, which impacts response time (albeit minimally with proper vectorization).

## 7. Future Work (Phase 3 Prep)
The upcoming Phase 3 involves:
1.  Validating this Risk-Aware Autoscaler architecture on the multi-region **Huawei workload dataset** to evaluate geographic model stability.
2.  **Sensitivity Pareto Analysis**: Plotting the continuum of the objective function parameter $\alpha \in [0.01, 0.20]$ to definitively map the efficient boundary of Idle Cost vs Extreme SLA.
