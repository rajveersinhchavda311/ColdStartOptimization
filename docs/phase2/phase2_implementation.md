# Phase 2: Implementation Details

## File Structure

The Phase 2 risk-aware logic is fully modularized and encapsulated within the following critical paths:
*   `models/risk_aware.py`: Contains the core prediction scaling logic, including GBR model training, GPD parameter fitting, scenario sampling, and CVaR optimization.
*   `models/baselines.py`: A library of baseline methodologies (Reactive, Static, MLP) ensuring fair evaluation.
*   `scripts/generate_phase2_results.py`: The unified execution pipeline that trains the models, simulates real-time deployment, calculates reliability metrics, and generates all 7 publication plots.

## Data Flow & Function Breakdown

The `RiskAwareScaler.predict(features)` pipeline executes synchronously per timestep during the rolling evaluation.

### 1. The P-Quantile Prediction
*   **Function**: `train_quantile_models()`
*   Three separate `GradientBoostingRegressor` instances are trained using the `'quantile'` loss function on the 10-lag autoregressive features. 
*   This establishes the lower (Q50), intermediate (Q90), and catastrophic (Q99) boundaries for the specific timestep.

### 2. Tail Parameter Extraction
*   **Function**: `fit_evt_model()`
*   The system uses the historical training data to find `Threshold = np.percentile(99)`.
*   A `scipy.stats.genpareto` distribution is fit to all exceedances ($y - threshold$).
*   If we lacked data to calculate $\xi$ and $\sigma$, we would fall back to normal assumptions, but the 12,000+ sample training blocks ensure high robustness here.

### 3. Scenario Matrix Construction
*   **Function**: `generate_scenarios()`
*   **Step**: Calculate EVT Gating: `use_evt = q99 > u`.
*   The system initializes an array of $N=300$ potential future loads.
*   If EVT is gated **OFF**, 100% of these 300 instances are derived from a Normal distribution $N(\mu=Q50, \sigma=\frac{Q90-Q50}{1.2816})$.
*   If EVT is gated **ON**, 3% of the samples are randomly drawn from the fitted heavy-tail GPD distribution.

### 4. Vectorized CVaR Resolution
*   **Function**: `optimize_cvar()`
*   **Search Bounds**: We compute the optimization across candidate $c$-values bounded to the realistic range: `[Q50, int(1.10 * Q99)]`.
*   A vectorized cost grid is computed (`C x S matrix`, testing all candidate container states against all 300 scenarios).
*   The conditional cost focuses strictly on the 1st percentile of worst-case samples (`alpha=0.01`). The candidate container state $c^*$ minimizing this top 1% mean is returned to scale the infrastructure.
