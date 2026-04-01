"""
Risk-Aware Autoscaler: EVT + CVaR
==================================
Combines quantile forecasting, extreme value theory, and conditional
value-at-risk optimization for tail-risk-aware container provisioning.

Architecture:
    1. Quantile Forecaster  — GBR with native pinball loss → (q50, q90, q99)
    2. EVT Tail Model       — GPD fitted to training exceedances above P99
    3. Scenario Generator   — Normal body + GPD tail → demand samples
    4. CVaR Optimizer       — grid search for min-CVaR container count

Integration:
    RiskAwareScaler.predict(features) → container count
    Plugs directly into evaluation.compare_models() without modification.
"""

import numpy as np
from scipy.stats import genpareto
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================================
# COMPONENT 1: QUANTILE FORECASTER
# ============================================================================

def train_quantile_models(train_data, quantiles=(0.50, 0.90, 0.99),
                          random_state=42, verbose=True):
    """
    Train one GradientBoostingRegressor per quantile with native pinball loss.

    INPUT:
        train_data : DataFrame with ['concurrency', 'lag_1', ..., 'lag_10']
        quantiles  : tuple of quantile levels to predict
        random_state : int, for reproducibility

    OUTPUT:
        models : dict {quantile_level: fitted GBR model}
    """
    feature_cols = [f'lag_{k}' for k in range(1, 11)]
    X = train_data[feature_cols].values
    y = train_data['concurrency'].values

    assert X.shape[1] == 10, "Expected 10 lag features"
    assert not np.any(np.isnan(X)), "NaN in features"
    assert not np.any(np.isnan(y)), "NaN in targets"

    models = {}

    for tau in quantiles:
        if verbose:
            print(f"  Training quantile model (tau={tau})...", end=" ", flush=True)

        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=tau,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=random_state,
        )
        model.fit(X, y)
        models[tau] = model

        if verbose:
            train_pred = model.predict(X)
            mae = np.mean(np.abs(train_pred - y))
            print(f"done (train MAE: {mae:,.0f})")

    # Validate quantile ordering on training data
    if verbose:
        preds = {tau: models[tau].predict(X) for tau in quantiles}
        sorted_q = sorted(quantiles)
        violations = 0
        for i in range(len(sorted_q) - 1):
            lower = preds[sorted_q[i]]
            upper = preds[sorted_q[i + 1]]
            violations += (lower > upper).sum()
        print(f"  Quantile ordering violations on train: {violations} "
              f"({100*violations/len(X)/max(1,len(sorted_q)-1):.1f}%)")

    return models


# ============================================================================
# COMPONENT 2: EVT TAIL MODEL (GPD)
# ============================================================================

def fit_evt_model(train_concurrency, threshold_percentile=99, verbose=True):
    """
    Fit Generalized Pareto Distribution to exceedances above threshold.

    INPUT:
        train_concurrency : array-like, training demand values
        threshold_percentile : int, percentile for threshold (default 99)

    OUTPUT:
        evt_params : dict with keys:
            'threshold'    : float, P99 of training data
            'shape'        : float, GPD shape parameter (xi)
            'scale'        : float, GPD scale parameter (sigma)
            'n_exceedances': int, number of exceedances used for fitting

    VALIDATES:
        - Sufficient exceedances (>= 10)
        - Shape parameter in reasonable range (-0.5, 1.0)
        - Scale parameter > 0
    """
    values = np.asarray(train_concurrency, dtype=float)
    threshold = np.percentile(values, threshold_percentile)

    # Extract exceedances
    exceedances = values[values > threshold] - threshold
    n_exc = len(exceedances)

    assert n_exc >= 10, (
        f"Too few exceedances ({n_exc}) for stable GPD fit. "
        f"Threshold={threshold:.0f}, need at least 10."
    )

    # Fit GPD (fix location=0 since exceedances are already shifted)
    shape, _, scale = genpareto.fit(exceedances, floc=0)

    # Validate parameters
    assert -0.5 < shape < 1.0, (
        f"GPD shape parameter out of range: xi={shape:.4f}. "
        f"Expected -0.5 < xi < 1.0 for workload data."
    )
    assert scale > 0, f"GPD scale must be positive, got {scale:.4f}"

    evt_params = {
        'threshold': float(threshold),
        'shape': float(shape),
        'scale': float(scale),
        'n_exceedances': int(n_exc),
    }

    if verbose:
        print(f"\n  EVT Model (GPD) fitted:")
        print(f"    Threshold (P{threshold_percentile}): {threshold:,.0f}")
        print(f"    Exceedances: {n_exc}")
        print(f"    Shape (xi):  {shape:.4f}")
        print(f"    Scale (sigma): {scale:,.2f}")
        # Sanity: expected value of GPD = sigma / (1 - xi) if xi < 1
        if shape < 1.0:
            expected_excess = scale / (1.0 - shape)
            print(f"    Expected excess above threshold: {expected_excess:,.0f}")

    return evt_params


# ============================================================================
# COMPONENT 3: SCENARIO GENERATION
# ============================================================================

def generate_scenarios(q50, q90, q99, evt_params, n_scenarios=300,
                       rng=None, use_evt=False):
    """
    Generate demand scenarios combining Normal body + optional GPD tail.

    LOGIC:
        If use_evt=True: Body (97%), Tail (3%) samples from GPD
        If use_evt=False: Body (100%)

    INPUT:
        q50, q90, q99 : float, predicted quantiles
        evt_params     : dict from fit_evt_model()
        n_scenarios    : int, total number of scenarios
        rng            : numpy Generator (for reproducibility in testing)

    OUTPUT:
        scenarios : array of shape (n_scenarios,), demand samples
    """
    if rng is None:
        rng = np.random.default_rng()

    if use_evt:
        tail_fraction = 0.03  # 3% tail
        n_tail = max(1, int(tail_fraction * n_scenarios))
    else:
        n_tail = 0
        
    n_body = n_scenarios - n_tail

    # --- Body: Normal distribution fitted to q50, q90 ---
    mu = max(0, q50)
    sigma = (q90 - q50) / 1.2816  # z-score of 90th percentile

    # Edge case: degenerate spread (q90 <= q50)
    if sigma <= 0:
        sigma = max(1.0, 0.1 * abs(mu))

    body_samples = rng.normal(mu, sigma, size=n_body)
    body_samples = np.clip(body_samples, 0, None)  # demand >= 0

    # --- Tail: GPD samples above threshold ---
    if use_evt and n_tail > 0:
        u = evt_params['threshold']
        xi = evt_params['shape']
        sig = evt_params['scale']

        # scipy genpareto: sample exceedances, then add threshold
        tail_exceedances = genpareto.rvs(xi, scale=sig, size=n_tail,
                                         random_state=rng.integers(0, 2**31))
        tail_samples = u + tail_exceedances

        # Combine
        scenarios = np.concatenate([body_samples, tail_samples])
    else:
        scenarios = body_samples

    return scenarios


# ============================================================================
# COMPONENT 4: CVaR OPTIMIZATION
# ============================================================================

def optimize_cvar(scenarios, q50, q99, alpha=0.01, cost_cold=10.0, cost_idle=1.0,
                  n_candidates=200):
    """
    Choose optimal container count c* minimizing CVaR.

    CVaR at level alpha = expected cost in the worst alpha fraction of scenarios.

    INPUT:
        scenarios    : array of demand samples
        alpha        : float, tail probability (0.05 = worst 5%)
        cost_cold    : float, penalty per unserved request
        cost_idle    : float, cost per idle container
        n_candidates : int, grid resolution for container count search

    OUTPUT:
        c_star : int, optimal container count
    """
    scenarios = np.asarray(scenarios, dtype=float)

    # Search space: bounds based on quantile forecast
    c_min = max(0, int(q50))
    
    # 10% margin above Q99 to bound search space
    margin = max(1, 0.10 * q99)
    c_max = int(q99 + margin) + 1

    if c_max <= c_min:
        return max(0, c_min)

    candidates = np.linspace(c_min, c_max, n_candidates).astype(int)

    # Number of worst-case scenarios to average (CVaR tail)
    k = max(1, int(alpha * len(scenarios)))

    # Vectorized cost computation: (n_candidates × n_scenarios)
    C = candidates[:, None]
    S = scenarios[None, :]
    cold = np.maximum(0, S - C)
    idle = np.maximum(0, C - S)
    costs = cost_idle * idle + cost_cold * cold

    # CVaR = mean of top-k costs per candidate (sorted descending)
    sorted_costs = np.sort(costs, axis=1)  # ascending
    cvar = sorted_costs[:, -k:].mean(axis=1)  # top-k = worst cases

    # Select optimal
    best_idx = np.argmin(cvar)
    c_star = int(candidates[best_idx])

    return max(0, c_star)


# ============================================================================
# WRAPPER: RiskAwareScaler
# ============================================================================

class RiskAwareScaler:
    """
    Risk-aware autoscaler: Quantile MLP + EVT + CVaR.

    Implements predict(features) for seamless integration with
    the evaluation framework. No modifications to evaluation/ needed.

    PIPELINE (per timestep):
        1. Predict quantiles (q50, q90, q99) from lag features
        2. Generate demand scenarios (Normal body + GPD tail)
        3. Optimize container count via CVaR minimization
        4. Return c* (integer container count)
    """

    def __init__(self, quantile_models, evt_params, alpha=0.05,
                 n_scenarios=300, cost_cold=10.0, cost_idle=1.0):
        """
        INPUT:
            quantile_models : dict {tau: fitted_model} from train_quantile_models()
            evt_params      : dict from fit_evt_model()
            alpha           : float, CVaR tail level (default 0.01 = worst 1%)
            n_scenarios     : int, scenarios per timestep (default 300)
            cost_cold       : float, cold start penalty
            cost_idle       : float, idle container cost
        """
        self.q_models = quantile_models
        self.evt_params = evt_params
        self.alpha = alpha
        self.n_scenarios = n_scenarios
        self.cost_cold = cost_cold
        self.cost_idle = cost_idle

        # Fixed RNG seed for reproducibility within a run
        self.rng = np.random.default_rng(seed=42)

    def predict(self, features):
        """
        Predict optimal container count for next timestep.

        INPUT: features, array of shape (10,) — [lag_1, ..., lag_10]
        OUTPUT: int, container count c*
        """
        features_2d = np.asarray(features, dtype=float).reshape(1, -1)

        # Step 1: Quantile predictions
        q50 = float(self.q_models[0.50].predict(features_2d)[0])
        q90 = float(self.q_models[0.90].predict(features_2d)[0])
        q99 = float(self.q_models[0.99].predict(features_2d)[0])

        # Enforce monotonicity
        q90 = max(q90, q50)
        q99 = max(q99, q90)

        # Step 1.5: EVT Gating
        threshold = self.evt_params['threshold']
        use_evt = (q99 > threshold)

        # Step 2: Generate scenarios
        scenarios = generate_scenarios(
            q50, q90, q99, self.evt_params,
            n_scenarios=self.n_scenarios,
            rng=self.rng,
            use_evt=use_evt
        )

        # Step 3: CVaR optimization
        c_star = optimize_cvar(
            scenarios,
            q50=q50,
            q99=q99,
            alpha=self.alpha,
            cost_cold=self.cost_cold,
            cost_idle=self.cost_idle,
        )

        return c_star
