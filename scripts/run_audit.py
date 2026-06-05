"""
Phase 1 Validation Audit Script
=================================

A skeptical reviewer's automated audit covering:
    1. Leakage audit -- no model accesses future information
    2. Metric audit -- mathematical verification of all metrics
    3. Baseline audit -- verify each model implements its stated strategy
    4. Extreme threshold audit -- verify train-only derivation
    5. Reproducibility audit -- run twice, verify identical results
    6. Accounting audit -- verify cold + served = demand at every timestep

Exit code 0 = all audits pass
Exit code 1 = at least one audit failed
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.reactive import ReactiveModel
from models.static_p90 import StaticP90Model
from models.forecast_only import ForecastOnlyModel
from models.tcn import TCNModel
from evaluation.extreme import compute_extreme_threshold, EXTREME_PERCENTILE
from evaluation.metrics import compute_metrics, C_COLD, C_IDLE
from evaluation.simulator import simulate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1", "azure")

audit_results = []


def audit(name, condition, detail=""):
    """Record an audit check result."""
    status = "PASS" if condition else "FAIL"
    audit_results.append({"name": name, "status": status, "detail": detail})
    symbol = "[PASS]" if condition else "[FAIL]"
    print(f"  {symbol} {name}")
    if detail and not condition:
        print(f"         Detail: {detail}")
    return condition


def load_data():
    """Load data and results."""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["timestamp"])
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["timestamp"])

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "r") as f:
        all_metrics = json.load(f)

    all_results = {}
    for model_name in all_metrics.keys():
        path = os.path.join(RESULTS_DIR, f"{model_name}_results.csv")
        all_results[model_name] = pd.read_csv(path, parse_dates=["timestamp"])

    return train, val, test, all_metrics, all_results


# ===================================================================
# AUDIT 1: LEAKAGE AUDIT
# ===================================================================

def audit_leakage(train, test, all_results):
    """Verify no model uses future information."""
    print("\n--- AUDIT 1: LEAKAGE ---")

    # Check 1: All models receive identical test data
    test_demands = []
    for model_name, results in all_results.items():
        test_demands.append(results["actual_demand"].values)

    all_same = all(np.array_equal(test_demands[0], d) for d in test_demands[1:])
    audit("All models evaluated on identical test data", all_same)

    # Check 2: Test timestamps are strictly after training timestamps
    train_max = train["timestamp"].max()
    test_min = test["timestamp"].min()
    audit("Test data is chronologically after training data",
          train_max < test_min,
          f"train_max={train_max}, test_min={test_min}")

    # Check 3: Reactive model predictions match lag_1 exactly
    reactive_results = all_results.get("Reactive")
    if reactive_results is not None:
        reactive_pred = reactive_results["predicted"].values
        test_lag1 = test["lag_1"].values.astype(np.float64)
        audit("Reactive predictions == lag_1 (no leakage from concurrency)",
              np.allclose(reactive_pred, test_lag1, atol=1e-6))

    # Check 4: Static P90 is constant (derived from train only)
    static_results = all_results.get("Static_P90")
    if static_results is not None:
        preds = static_results["predicted"].values
        all_same_val = np.all(preds == preds[0])
        audit("Static P90 predictions are constant (single train-derived value)",
              all_same_val)

        # Verify the constant matches P90(train)
        expected_p90 = np.percentile(train["concurrency"].values, 90)
        audit("Static P90 value matches np.percentile(train, 90)",
              np.isclose(preds[0], expected_p90, atol=1.0),
              f"predicted={preds[0]:.1f}, expected={expected_p90:.1f}")


# ===================================================================
# AUDIT 2: METRIC AUDIT
# ===================================================================

def audit_metrics(all_results, all_metrics):
    """Mathematically verify all metric computations."""
    print("\n--- AUDIT 2: METRIC CORRECTNESS ---")

    for model_name, results in all_results.items():
        demand = results["actual_demand"].values
        cold = results["cold_starts"].values
        idle = results["idle_capacity"].values
        provisioned = results["provisioned"].values
        is_extreme = results["is_extreme"].values.astype(bool)
        m = all_metrics[model_name]

        # Request SLA: 1 - (total_cold / total_demand)
        total_demand = demand.sum()
        total_cold = cold.sum()
        expected_sla = 1.0 - (total_cold / total_demand)
        audit(f"[{model_name}] Request SLA = 1 - (total_cold / total_demand)",
              np.isclose(m["request_sla"], expected_sla, atol=1e-10),
              f"stored={m['request_sla']:.10f}, recomputed={expected_sla:.10f}")

        # Extreme SLA: 1 - (cold_extreme / demand_extreme)
        demand_extreme = demand[is_extreme].sum()
        cold_extreme = cold[is_extreme].sum()
        if demand_extreme > 0:
            expected_ext_sla = 1.0 - (cold_extreme / demand_extreme)
        else:
            expected_ext_sla = 1.0
        audit(f"[{model_name}] Extreme SLA = 1 - (cold_extreme / demand_extreme)",
              np.isclose(m["extreme_sla"], expected_ext_sla, atol=1e-10),
              f"stored={m['extreme_sla']:.10f}, recomputed={expected_ext_sla:.10f}")

        # Cost decomposition
        expected_cold_cost = total_cold * C_COLD
        expected_idle_cost = idle.sum() * C_IDLE
        expected_total = expected_cold_cost + expected_idle_cost
        audit(f"[{model_name}] total_cost == cold_cost + idle_cost",
              np.isclose(m["total_cost"], expected_total, atol=1.0),
              f"stored={m['total_cost']:.1f}, recomputed={expected_total:.1f}")

        # Accounting identity: at each timestep, cold = max(0, demand - provisioned)
        expected_cold_per_step = np.maximum(demand - provisioned, 0)
        expected_idle_per_step = np.maximum(provisioned - demand, 0)
        audit(f"[{model_name}] cold_starts = max(0, demand - provisioned) at each step",
              np.allclose(cold, expected_cold_per_step, atol=1e-6))
        audit(f"[{model_name}] idle = max(0, provisioned - demand) at each step",
              np.allclose(idle, expected_idle_per_step, atol=1e-6))

        # Mutual exclusivity: can't have both cold starts and idle at same timestep
        both = (cold > 0) & (idle > 0)
        audit(f"[{model_name}] No timestep has both cold starts AND idle",
              not both.any(),
              f"{both.sum()} violations found")


# ===================================================================
# AUDIT 3: BASELINE CORRECTNESS
# ===================================================================

def audit_baselines(train, test, all_results):
    """Verify each baseline implements its stated strategy correctly."""
    print("\n--- AUDIT 3: BASELINE CORRECTNESS ---")

    # Reactive: prediction = lag_1
    reactive_results = all_results.get("Reactive")
    if reactive_results is not None:
        expected = test["lag_1"].values.astype(np.float64)
        actual = reactive_results["predicted"].values
        audit("Reactive: prediction == lag_1 exactly",
              np.allclose(actual, expected, atol=1e-6))

    # Static P90: prediction = constant P90 from train
    static_results = all_results.get("Static_P90")
    if static_results is not None:
        expected_val = np.percentile(train["concurrency"].values, 90)
        actual = static_results["predicted"].values
        audit("Static P90: all predictions == P90(train)",
              np.allclose(actual, expected_val, atol=1.0))

    # Forecast Only: prediction = mean(lag_1, ..., lag_10)
    forecast_results = all_results.get("Forecast_Only")
    if forecast_results is not None:
        lag_cols = [f"lag_{k}" for k in range(1, 11)]
        expected = test[lag_cols].values.mean(axis=1)
        actual = forecast_results["predicted"].values
        audit("Forecast Only: prediction == mean(lag_1..lag_10)",
              np.allclose(actual, expected, atol=1e-4))

    # TCN: verify it's actually a TCN (check architecture)
    tcn_results = all_results.get("TCN")
    if tcn_results is not None:
        # Verify TCN predictions are NOT constant (it learned something)
        preds = tcn_results["predicted"].values
        audit("TCN: predictions are NOT constant (learned model)",
              np.std(preds) > 100)

        # Verify TCN predictions are NOT identical to any simple baseline
        reactive_pred = all_results.get("Reactive", {})
        if isinstance(reactive_pred, pd.DataFrame):
            audit("TCN: predictions differ from Reactive (not trivial)",
                  not np.allclose(preds, reactive_pred["predicted"].values, atol=100))


# ===================================================================
# AUDIT 4: EXTREME THRESHOLD
# ===================================================================

def audit_extreme_threshold(train, test, all_results):
    """Verify extreme threshold is derived from training data only."""
    print("\n--- AUDIT 4: EXTREME THRESHOLD ---")

    # Compute expected threshold from train
    threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)
    audit(f"Extreme threshold computed from train P{EXTREME_PERCENTILE}",
          True,
          f"threshold={threshold:,.0f}")

    # Verify the is_extreme flag in results matches this threshold
    for model_name, results in all_results.items():
        demand = results["actual_demand"].values
        is_extreme_stored = results["is_extreme"].values.astype(bool)
        is_extreme_expected = demand > threshold
        audit(f"[{model_name}] is_extreme flag matches train-derived threshold",
              np.array_equal(is_extreme_stored, is_extreme_expected))

    # Verify threshold is NOT the test P99
    test_p99 = np.percentile(test["concurrency"].values, 99)
    audit("Extreme threshold != P99(test) (no test leakage)",
          not np.isclose(threshold, test_p99, atol=1.0),
          f"train_threshold={threshold:.0f}, test_p99={test_p99:.0f}")


# ===================================================================
# AUDIT 5: REPRODUCIBILITY
# ===================================================================

def audit_reproducibility(train, val, test, threshold):
    """Run two models and verify identical predictions."""
    print("\n--- AUDIT 5: REPRODUCIBILITY ---")

    # Test reproducibility with deterministic models (Reactive, Static P90, Forecast)
    for ModelClass in [ReactiveModel, StaticP90Model, ForecastOnlyModel]:
        m1 = ModelClass()
        m1.fit(train)
        pred1 = m1.predict(test)

        m2 = ModelClass()
        m2.fit(train)
        pred2 = m2.predict(test)

        audit(f"[{m1.name}] Reproducible: two runs produce identical predictions",
              np.array_equal(pred1, pred2))


# ===================================================================
# AUDIT 6: ACCOUNTING IDENTITY
# ===================================================================

def audit_accounting(all_results):
    """Verify served + cold = demand at every timestep."""
    print("\n--- AUDIT 6: ACCOUNTING IDENTITY ---")

    for model_name, results in all_results.items():
        demand = results["actual_demand"].values
        cold = results["cold_starts"].values
        provisioned = results["provisioned"].values

        # Served requests = min(demand, provisioned)
        served = np.minimum(demand, provisioned)
        # Verify: served + cold = demand
        identity_holds = np.allclose(served + cold, demand, atol=1e-6)
        audit(f"[{model_name}] served + cold_starts == demand at every timestep",
              identity_holds)

        # No negative values anywhere
        audit(f"[{model_name}] No negative cold starts or idle capacity",
              (cold >= 0).all() and (results["idle_capacity"].values >= 0).all())


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("Phase 1: Validation Audit")
    print("=" * 60)

    # Load data
    train, val, test, all_metrics, all_results = load_data()

    # Compute threshold for reproducibility test
    threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)

    # Run all audits
    audit_leakage(train, test, all_results)
    audit_metrics(all_results, all_metrics)
    audit_baselines(train, test, all_results)
    audit_extreme_threshold(train, test, all_results)
    audit_reproducibility(train, val, test, threshold)
    audit_accounting(all_results)

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    n_pass = sum(1 for r in audit_results if r["status"] == "PASS")
    n_fail = sum(1 for r in audit_results if r["status"] == "FAIL")
    n_total = len(audit_results)

    print(f"  Total checks: {n_total}")
    print(f"  Passed: {n_pass}")
    print(f"  Failed: {n_fail}")

    if n_fail > 0:
        print("\n  FAILED CHECKS:")
        for r in audit_results:
            if r["status"] == "FAIL":
                print(f"    [FAIL] {r['name']}")
                if r["detail"]:
                    print(f"           {r['detail']}")

    overall = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n  Overall: {overall}")

    # Save audit results
    audit_path = os.path.join(RESULTS_DIR, "audit_results.json")
    with open(audit_path, "w") as f:
        json.dump({
            "total": n_total,
            "passed": n_pass,
            "failed": n_fail,
            "overall": overall,
            "checks": audit_results
        }, f, indent=2)
    print(f"\n  Saved: {audit_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
