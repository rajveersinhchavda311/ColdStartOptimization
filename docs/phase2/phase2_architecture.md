# Phase 2: Risk-Aware Autoscaling Architecture

## 1. Problem Statement: The Failure of Standard Forecasting
Standard machine learning models (e.g., Temporal Convolutional Networks, LSTMs) optimize for the expected value (mean squared error or mean absolute error). In serverless cloud environments, demand distribution is highly skewed and features massive, unpredictable spikes. 

When a "standard" forecaster encounters an extreme spike (e.g., P99 demand), it overwhelmingly under-predicts. Because the penalty for a cold start is an order of magnitude higher than the cost of an idle container ($C_{cold} = 10 \times C_{idle}$), these periodic SLA violations destroy the economic viability of the system during volatile regimes.

## 2. Core Methodological Shift
The Risk-Aware Autoscaler discards point-forecasting in favor of **probabilistic tail-risk mitigation**. Instead of asking "What is the most likely demand?", it asks "How much capacity must we deploy to cap our expected penalty in the worst-case scenarios?"

This is achieved via a pipeline of three statistical components:
1. **Quantile Forecasting:** Establishing boundaries for normal variance.
2. **Extreme Value Theory (EVT):** Modeling the statistically "unseen" heavy tail.
3. **Conditional Value-at-Risk (CVaR):** Optimizing the capacity provision against a cost function.

---

## 3. Pipeline Components

### 3.1 Quantile Forecasting (Gradient Boosting)
We replace the single-point prediction with three quantiles ($P_{50}, P_{90}, P_{99}$) using `GradientBoostingRegressor`.
*   **Math Intuition**: By training on the pinball loss function, $L_\tau(y, \hat{y}) = \max(\tau(y-\hat{y}), (1-\tau)(\hat{y}-y))$, the model explicitly learns the upper bounds of demand variance conditioned on the recent lag features.
*   **Role**: Defines the "body" of the demand distribution (the 97% of normal operations).

### 3.2 Extreme Value Theory (Generalized Pareto Distribution)
Traditional distributions (Normal, Log-Normal) decay too quickly and exponentially underestimate the probability of severe spikes. We apply Extreme Value Theory (specifically the Peaks-Over-Threshold method) to model the tail.
*   **Threshold ($u$)**: The 99th percentile of the training set.
*   **GPD Fit**: We fit a Generalized Pareto Distribution to all training exceedances above $u$, learning a shape ($\xi$) and scale ($\sigma$) parameter.
*   **Role**: Generates synthetic "Black Swan" demand scenarios that the base model has never seen.

### 3.3 Conditional Value-at-Risk (CVaR) Optimization
At each timestep, we generate $N=300$ potential future realities (scenarios).
Let $C$ be the candidate container count, and $S_i$ be the $i$-th demand scenario. The cost is calculated as:
$$ Cost(C, S_i) = C_{idle} \times \max(0, C - S_i) + C_{cold} \times \max(0, S_i - C) $$

We calculate the **CVaR at $\alpha=0.01$**, which is the expected cost in the worst 1% of the generated scenarios. The model selects the candidate $c^*$ that minimizes this CVaR.

## 4. Conditional Gating: The "Risk Gate"
V1 of this model ran EVT injections continuously, resulting in permanent over-provisioning (a 1.28x provisioning ratio). 

To ensure **adaptive intelligence**, V2 implements conditional gating:
$$ \text{IF } Q99_{forecast} \ge u_{train} \implies \text{Activate EVT} $$

If the local P99 forecast breaches the global historical extreme threshold, the model generates 3% of its scenarios from the GPD tail. If the local P99 is safe, the optimizer is fed only standard Normal scenarios. This allows the model to act as a lean, efficient forecaster during calm traffic and an impenetrable shield during storms.
