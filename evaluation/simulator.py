"""
Timestep-by-Timestep Provisioning Simulator
=============================================

This module simulates the provisioning process for a serverless platform,
evaluating a model's predictions against actual demand at each timestep.

Design:
    For each timestep t:
        1. Model produces prediction p[t] using ONLY historical lag features
        2. Actual demand is d[t] = concurrency[t]
        3. Provisioned capacity = ceil(p[t])  (can't provision fractional slots)
        4. Cold starts = max(0, d[t] - provisioned[t])
        5. Idle capacity = max(0, provisioned[t] - d[t])
        6. Cost = cold_starts × c_cold + idle × c_idle

    Provisioning policy: provision = prediction (Phase 1)
    This is deliberately simple — we want to answer whether forecasting
    alone is sufficient for autoscaling before introducing risk-aware logic.

Leakage prevention:
    - The model.predict() method receives the full DataFrame but is
      contractually required to use ONLY lag_* columns
    - The simulator itself only reads 'concurrency' AFTER predictions are made
    - No feedback loop: predictions at time t cannot depend on outcomes at t

Output:
    A DataFrame with per-timestep breakdown:
    [timestamp, actual_demand, predicted, provisioned,
     cold_starts, idle_capacity, cost, is_extreme]
"""

import numpy as np
import pandas as pd
from evaluation.extreme import flag_extreme_events


def simulate(model, eval_df: pd.DataFrame, extreme_threshold: float,
             c_cold: float = 10, c_idle: float = 1) -> pd.DataFrame:
    """
    Run timestep-by-timestep provisioning simulation.

    Parameters
    ----------
    model : BaseModel
        A fitted model implementing the BaseModel interface.
    eval_df : pd.DataFrame
        Evaluation data with columns: timestamp, concurrency, lag_1..lag_10
    extreme_threshold : float
        Threshold for extreme events (computed from TRAINING data).
    c_cold : float
        Cost per cold-started request (default: 10).
    c_idle : float
        Cost per idle provisioned slot (default: 1).

    Returns
    -------
    pd.DataFrame
        Per-timestep results with columns:
        [timestamp, actual_demand, predicted, provisioned,
         cold_starts, idle_capacity, cost, is_extreme]
    """
    # Step 1: Generate predictions using only lag features
    # The model.predict() method is contractually required to use only lag_* columns
    predictions = model.predict(eval_df)

    # Step 2: Extract actual demand
    actual_demand = eval_df["concurrency"].values.astype(np.float64)

    # Step 3: Compute provisioned capacity
    # Use ceil() because we can't provision fractional slots
    # Also clip to >= 0 (can't provision negative capacity)
    provisioned = np.maximum(np.ceil(predictions), 0).astype(np.float64)

    # Step 4: Compute cold starts and idle capacity
    cold_starts = np.maximum(actual_demand - provisioned, 0)
    idle_capacity = np.maximum(provisioned - actual_demand, 0)

    # Step 5: Compute cost
    cost = cold_starts * c_cold + idle_capacity * c_idle

    # Step 6: Flag extreme events using train-derived threshold
    is_extreme = flag_extreme_events(actual_demand, extreme_threshold)

    # Build results DataFrame
    results_df = pd.DataFrame({
        "timestamp": eval_df["timestamp"].values,
        "actual_demand": actual_demand,
        "predicted": predictions,
        "provisioned": provisioned,
        "cold_starts": cold_starts,
        "idle_capacity": idle_capacity,
        "cost": cost,
        "is_extreme": is_extreme,
    })

    # Sanity checks
    _validate_results(results_df)

    return results_df


def _validate_results(results_df: pd.DataFrame) -> None:
    """
    Run sanity checks on simulation results.

    These are invariants that must hold regardless of the model or data.
    """
    cold = results_df["cold_starts"].values
    idle = results_df["idle_capacity"].values
    demand = results_df["actual_demand"].values
    provisioned = results_df["provisioned"].values

    # 1. No negative values
    assert (cold >= 0).all(), "Negative cold starts found"
    assert (idle >= 0).all(), "Negative idle capacity found"
    assert (demand >= 0).all(), "Negative demand found"
    assert (provisioned >= 0).all(), "Negative provisioned capacity found"

    # 2. Cold starts + idle can't both be positive at same timestep
    both = (cold > 0) & (idle > 0)
    assert not both.any(), f"{both.sum()} timesteps have both cold starts and idle"

    # 3. Accounting identity: provisioned = demand - cold + idle
    # Equivalently: cold = demand - provisioned + idle (when demand > provisioned)
    # More precisely: cold = max(0, demand - provisioned) and idle = max(0, provisioned - demand)
    expected_cold = np.maximum(demand - provisioned, 0)
    expected_idle = np.maximum(provisioned - demand, 0)
    assert np.allclose(cold, expected_cold, atol=1e-6), "Cold start accounting mismatch"
    assert np.allclose(idle, expected_idle, atol=1e-6), "Idle capacity accounting mismatch"

    # 4. No NaN/Inf values
    for col in ["actual_demand", "predicted", "provisioned", "cold_starts",
                "idle_capacity", "cost"]:
        vals = results_df[col].values
        assert not np.isnan(vals).any(), f"NaN found in {col}"
        assert not np.isinf(vals).any(), f"Inf found in {col}"
