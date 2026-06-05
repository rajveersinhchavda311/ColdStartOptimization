"""
Extreme Event Analysis Module
===============================

LEAKAGE GUARD:
    The extreme threshold is computed EXCLUSIVELY from training data.
    It is NEVER derived from test data, validation data, or evaluation outputs.

    This is mathematically necessary: if we defined "extreme" using test data,
    we would be evaluating our model's performance on events whose definition
    depends on data the model never saw — this is circular and constitutes
    information leakage.

Methodology:
    extreme_threshold = P99(train_concurrency)

    An extreme event at time t is defined as:
        actual_demand[t] > extreme_threshold

    The P99 percentile is used (not P95) to focus on truly exceptional
    demand spikes. This is an experimental choice documented explicitly.
"""

import pandas as pd
import numpy as np


# Default percentile for extreme event definition
EXTREME_PERCENTILE = 99


def compute_extreme_threshold(train_df: pd.DataFrame,
                               percentile: int = EXTREME_PERCENTILE) -> float:
    """
    Compute the extreme event threshold from TRAINING data ONLY.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split with 'concurrency' column.
    percentile : int
        Percentile to use for threshold (default: 99 = P99).

    Returns
    -------
    float
        The threshold value. Demand exceeding this value is "extreme."

    Notes
    -----
    This function takes ONLY train_df as input. It has no access to
    validation or test data. This is by design — any modification to
    accept non-training data would constitute a leakage violation.
    """
    threshold = float(np.percentile(train_df["concurrency"].values, percentile))
    return threshold


def flag_extreme_events(demand: np.ndarray, threshold: float) -> np.ndarray:
    """
    Flag timesteps where demand exceeds the extreme threshold.

    Parameters
    ----------
    demand : np.ndarray
        Actual demand values for each timestep.
    threshold : float
        The extreme threshold (computed from training data).

    Returns
    -------
    np.ndarray[bool]
        Boolean mask — True where demand > threshold.
    """
    return demand > threshold


def summarize_extreme_events(demand: np.ndarray, threshold: float,
                              split_name: str = "unknown") -> dict:
    """
    Print and return summary statistics about extreme events.

    Parameters
    ----------
    demand : np.ndarray
        Actual demand values.
    threshold : float
        Extreme threshold (from training data).
    split_name : str
        Name of the data split for display purposes.

    Returns
    -------
    dict
        Summary statistics.
    """
    is_extreme = flag_extreme_events(demand, threshold)
    n_extreme = int(is_extreme.sum())
    n_total = len(demand)
    pct = 100.0 * n_extreme / n_total if n_total > 0 else 0.0

    summary = {
        "split": split_name,
        "threshold": threshold,
        "percentile_used": EXTREME_PERCENTILE,
        "total_timesteps": n_total,
        "extreme_timesteps": n_extreme,
        "extreme_pct": pct,
        "extreme_demand_mean": float(demand[is_extreme].mean()) if n_extreme > 0 else 0.0,
        "extreme_demand_max": float(demand[is_extreme].max()) if n_extreme > 0 else 0.0,
    }

    print(f"  Extreme events ({split_name}): {n_extreme}/{n_total} "
          f"({pct:.1f}%) timesteps exceed threshold {threshold:,.0f}")

    return summary
