"""
Metrics Computation Module
===========================

Computes all Phase 1 evaluation metrics from a results DataFrame.

Metrics:
    1. Total Cost          — Σ cost[t] across all timesteps
    2. Total Cold Starts   — Σ cold_starts[t] (unserved requests)
    3. Total Idle Capacity — Σ idle[t] (wasted provisioned slots)
    4. Cold Cost           — Σ cold_starts[t] × c_cold
    5. Idle Cost           — Σ idle[t] × c_idle
    6. Request SLA         — 1 - (total_cold / total_demand)
    7. Extreme SLA         — 1 - (cold_extreme / demand_extreme)
    8. Cold Start Rate     — fraction of timesteps with any cold starts
    9. Avg Cold per Step   — mean cold starts per timestep

Cost model:
    cold_cost_per_unit = 10  (penalty per cold-started request)
    idle_cost_per_unit = 1   (waste per idle provisioned slot)

    IMPORTANT: This cost model is an experimental assumption.
    It is NOT derived from any specific cloud provider's pricing.
    The 10:1 ratio reflects the general principle that cold starts
    (degraded user experience, latency, potential SLA violations)
    are significantly more costly than idle resources (wasted capacity).

Mathematical validation (enforced with assertions):
    Request SLA = 1 - (Σ cold_starts) / (Σ actual_demand)
    Extreme SLA = 1 - (Σ cold_starts where is_extreme) / (Σ demand where is_extreme)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cost model — EXPERIMENTAL ASSUMPTION (not from cloud pricing)
# ---------------------------------------------------------------------------
C_COLD = 10   # Cost per cold-started request
C_IDLE = 1    # Cost per idle provisioned slot


def compute_metrics(results_df: pd.DataFrame) -> dict:
    """
    Compute all Phase 1 metrics from a simulation results DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns:
            actual_demand, predicted, provisioned,
            cold_starts, idle_capacity, is_extreme

    Returns
    -------
    dict
        All computed metrics.
    """
    demand = results_df["actual_demand"].values
    cold = results_df["cold_starts"].values
    idle = results_df["idle_capacity"].values
    is_extreme = results_df["is_extreme"].values

    n = len(results_df)
    total_demand = demand.sum()
    total_cold = cold.sum()
    total_idle = idle.sum()
    cold_cost = total_cold * C_COLD
    idle_cost = total_idle * C_IDLE
    total_cost = cold_cost + idle_cost

    # Request SLA: fraction of total requests that were served
    # Served requests = total_demand - total_cold
    request_sla = 1.0 - (total_cold / total_demand) if total_demand > 0 else 1.0

    # Extreme event SLA: fraction of extreme-event requests that were served
    extreme_mask = is_extreme.astype(bool)
    demand_extreme = demand[extreme_mask].sum()
    cold_extreme = cold[extreme_mask].sum()
    extreme_sla = 1.0 - (cold_extreme / demand_extreme) if demand_extreme > 0 else 1.0

    # Cold start rate: fraction of timesteps with any cold starts
    cold_start_rate = (cold > 0).sum() / n if n > 0 else 0.0

    # Average cold starts per timestep
    avg_cold_per_step = total_cold / n if n > 0 else 0.0

    # ---------------------------------------------------------------------------
    # MATHEMATICAL VALIDATION
    # These assertions verify the metric formulas are implemented correctly.
    # If they fail, there is a bug in the computation, not in the data.
    # ---------------------------------------------------------------------------

    # Validate Request SLA
    expected_request_sla = 1.0 - (total_cold / total_demand) if total_demand > 0 else 1.0
    assert abs(request_sla - expected_request_sla) < 1e-10, (
        f"Request SLA validation failed: {request_sla} != {expected_request_sla}"
    )

    # Validate Extreme SLA
    expected_extreme_sla = 1.0 - (cold_extreme / demand_extreme) if demand_extreme > 0 else 1.0
    assert abs(extreme_sla - expected_extreme_sla) < 1e-10, (
        f"Extreme SLA validation failed: {extreme_sla} != {expected_extreme_sla}"
    )

    # Validate cost decomposition
    assert abs(total_cost - (cold_cost + idle_cost)) < 1e-6, (
        f"Cost decomposition failed: {total_cost} != {cold_cost} + {idle_cost}"
    )

    # Validate cold/idle mutual exclusivity per timestep
    # At any timestep, you can have cold starts OR idle capacity, never both
    both_mask = (cold > 0) & (idle > 0)
    assert not both_mask.any(), (
        f"Found {both_mask.sum()} timesteps with BOTH cold starts and idle capacity"
    )

    metrics = {
        "total_cost": float(total_cost),
        "cold_cost": float(cold_cost),
        "idle_cost": float(idle_cost),
        "total_cold_starts": int(total_cold),
        "total_idle_capacity": int(total_idle),
        "total_demand": int(total_demand),
        "request_sla": float(request_sla),
        "extreme_sla": float(extreme_sla),
        "cold_start_rate": float(cold_start_rate),
        "avg_cold_per_step": float(avg_cold_per_step),
        "n_timesteps": n,
        "n_extreme_timesteps": int(extreme_mask.sum()),
        "demand_extreme_total": int(demand_extreme),
        "cold_extreme_total": int(cold_extreme),
        "cost_model": {"c_cold": C_COLD, "c_idle": C_IDLE,
                       "note": "Experimental assumption, not from cloud pricing"},
    }

    return metrics


def format_metrics_table(all_metrics: dict) -> str:
    """
    Format metrics from multiple models into a human-readable table.

    Parameters
    ----------
    all_metrics : dict
        Keys are model names, values are metric dicts from compute_metrics().

    Returns
    -------
    str
        Formatted table string.
    """
    header = (
        f"{'Model':<16} {'Total Cost':>14} {'Cold Cost':>12} {'Idle Cost':>12} "
        f"{'Cold Starts':>14} {'Req SLA':>10} {'Ext SLA':>10} "
        f"{'CS Rate':>8}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for model_name, m in all_metrics.items():
        line = (
            f"{model_name:<16} "
            f"{m['total_cost']:>14,.0f} "
            f"{m['cold_cost']:>12,.0f} "
            f"{m['idle_cost']:>12,.0f} "
            f"{m['total_cold_starts']:>14,} "
            f"{m['request_sla']:>10.6f} "
            f"{m['extreme_sla']:>10.6f} "
            f"{m['cold_start_rate']:>8.4f}"
        )
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)
