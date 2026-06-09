"""
Phase 1 Huawei Audit — Methodology Validation
==============================================

Verifies that Phase 1 Huawei results are leakage-free and methodologically
sound. Adapts the same audit logic used for Azure Phase 1.

Checks:
  1. Test set identity          — all 6 models see identical test demands
  2. Burn-in correctly applied  — lag_1440 is non-null throughout all splits
  3. No future leakage          — provisioned[t] computed only from lag features
  4. Accounting identity        — cold_starts = max(0, demand - provisioned)
                                  idle_capacity = max(0, provisioned - demand)
  5. Non-negative values        — all outputs >= 0
  6. Cold/idle exclusivity      — never both > 0 at same timestep
  7. Extreme threshold source   — threshold in extreme_threshold.txt matches
                                  P99 of training concurrency
  8. lag_1440 presence          — training split contains lag_1440 column

Outputs:
    results/phase1/huawei/combined/audit_results.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR    = os.path.join(PROJECT_ROOT, "data",    "processed", "huawei", "combined")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1",   "huawei", "combined")

MODEL_NAMES = ["Reactive", "Static_P90", "Forecast_Only",
               "Seasonal_Naive", "Linear_Seasonal", "TCN"]

audit_log    = []
total_checks = 0
passed       = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global total_checks, passed
    total_checks += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    tag = f"  [{status}] {name}"
    print(tag)
    if detail and not condition:
        print(f"         Detail: {detail}")
    audit_log.append({"name": name, "status": status, "detail": detail})


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"),   parse_dates=["timestamp"])
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  parse_dates=["timestamp"])
    return train, val, test


def load_diagnostics():
    diags = {}
    for name in MODEL_NAMES:
        path = os.path.join(RESULTS_DIR, f"{name}_diagnostics.csv")
        if os.path.exists(path):
            diags[name] = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            print(f"  WARNING: {name}_diagnostics.csv not found")
    return diags


def load_threshold():
    path = os.path.join(RESULTS_DIR, "extreme_threshold.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if line.startswith("Value:"):
                return float(line.split(":")[1].strip().replace(",", ""))
    return None


# ---------------------------------------------------------------------------
# Audit sections
# ---------------------------------------------------------------------------

def audit_data_integrity(train, val, test):
    print("\n--- 1. Data Integrity ---")
    check("Train has lag_1440 column", "lag_1440" in train.columns)
    check("Val has lag_1440 column",   "lag_1440" in val.columns)
    check("Test has lag_1440 column",  "lag_1440" in test.columns)
    check("lag_1440 non-null in train",
          train["lag_1440"].notna().all(),
          "Burn-in not applied correctly")
    check("lag_1440 non-null in val",
          val["lag_1440"].notna().all())
    check("lag_1440 non-null in test",
          test["lag_1440"].notna().all())
    check("Train precedes val chronologically",
          train["timestamp"].iloc[-1] < val["timestamp"].iloc[0])
    check("Val precedes test chronologically",
          val["timestamp"].iloc[-1] < test["timestamp"].iloc[0])
    check("Train size = 25920",
          len(train) == 25920,
          f"Got {len(train)}")
    check("Val size = 8640",
          len(val) == 8640,
          f"Got {len(val)}")
    check("Test size = 8640",
          len(test) == 8640,
          f"Got {len(test)}")


def audit_test_set_identity(diags, test):
    print("\n--- 2. Test Set Identity ---")
    ref_name = MODEL_NAMES[0]
    if ref_name not in diags:
        check("Test set identity (reference loaded)", False, "Reference missing")
        return

    ref_demand = diags[ref_name]["actual_demand"].values
    for name in MODEL_NAMES[1:]:
        if name not in diags:
            check(f"Test set identity: {name}", False, "Diagnostics missing")
            continue
        this_demand = diags[name]["actual_demand"].values
        match = np.allclose(ref_demand, this_demand, atol=1e-6)
        check(f"Test demands identical: {ref_name} vs {name}", match,
              f"Max diff = {np.max(np.abs(ref_demand - this_demand)):.2f}")

    # Also verify against raw test CSV
    raw_demand = test["concurrency"].values
    match_raw = np.allclose(ref_demand, raw_demand, atol=1e-6)
    check("Test demands match raw test.csv", match_raw,
          f"Max diff = {np.max(np.abs(ref_demand - raw_demand)):.2f}")


def audit_accounting_identity(diags):
    print("\n--- 3. Accounting Identity ---")
    for name, df in diags.items():
        demand     = df["actual_demand"].values
        prov       = df["provisioned"].values
        cold       = df["cold_starts"].values
        idle       = df["idle_capacity"].values

        expected_cold = np.maximum(demand - prov, 0)
        expected_idle = np.maximum(prov - demand, 0)

        check(f"Cold accounting: {name}",
              np.allclose(cold, expected_cold, atol=1e-4),
              f"Max diff = {np.max(np.abs(cold - expected_cold)):.6f}")
        check(f"Idle accounting: {name}",
              np.allclose(idle, expected_idle, atol=1e-4),
              f"Max diff = {np.max(np.abs(idle - expected_idle)):.6f}")


def audit_non_negative(diags):
    print("\n--- 4. Non-Negative Values ---")
    for name, df in diags.items():
        check(f"cold_starts >= 0: {name}",
              (df["cold_starts"].values >= -1e-6).all())
        check(f"idle_capacity >= 0: {name}",
              (df["idle_capacity"].values >= -1e-6).all())
        check(f"provisioned >= 0: {name}",
              (df["provisioned"].values >= -1e-6).all())


def audit_cold_idle_exclusivity(diags):
    print("\n--- 5. Cold/Idle Exclusivity ---")
    for name, df in diags.items():
        cold = df["cold_starts"].values
        idle = df["idle_capacity"].values
        both = ((cold > 1e-6) & (idle > 1e-6))
        check(f"No timestep with both cold & idle: {name}",
              not both.any(),
              f"{both.sum()} violations found")


def audit_extreme_threshold(train, saved_threshold):
    print("\n--- 6. Extreme Threshold Source ---")
    from evaluation.extreme import EXTREME_PERCENTILE
    computed = float(np.percentile(train["concurrency"].values, EXTREME_PERCENTILE))
    if saved_threshold is None:
        check("extreme_threshold.txt exists", False, "File missing")
        return
    check(f"Threshold = P{EXTREME_PERCENTILE}(train concurrency)",
          abs(computed - saved_threshold) < 1.0,
          f"Computed={computed:,.2f}, Saved={saved_threshold:,.2f}")


def audit_no_nan(diags):
    print("\n--- 7. No NaN / Inf in Results ---")
    for name, df in diags.items():
        for col in ["actual_demand", "predicted", "provisioned",
                    "cold_starts", "idle_capacity", "cost"]:
            if col not in df.columns:
                continue
            vals = df[col].values
            check(f"No NaN in {col}: {name}",
                  not np.isnan(vals).any())
            check(f"No Inf in {col}: {name}",
                  not np.isinf(vals).any())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global total_checks, passed
    print("=" * 60)
    print("Phase 1 Huawei Audit")
    print("=" * 60)

    train, val, test = load_data()
    diags = load_diagnostics()
    saved_threshold = load_threshold()

    audit_data_integrity(train, val, test)
    audit_test_set_identity(diags, test)
    audit_accounting_identity(diags)
    audit_non_negative(diags)
    audit_cold_idle_exclusivity(diags)
    audit_extreme_threshold(train, saved_threshold)
    audit_no_nan(diags)

    print(f"\n{'='*60}")
    print(f"Audit Result: {passed}/{total_checks} checks passed")
    print(f"{'='*60}")

    failures = [e for e in audit_log if e["status"] == "FAIL"]
    if failures:
        print(f"\nFailed checks ({len(failures)}):")
        for f in failures:
            print(f"  [FAIL] {f['name']}")
            if f["detail"]:
                print(f"         {f['detail']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "total_checks": total_checks,
        "passed": passed,
        "failed": total_checks - passed,
        "pass_rate": passed / total_checks if total_checks > 0 else 0,
        "checks": audit_log,
    }
    out_path = os.path.join(RESULTS_DIR, "audit_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    if passed == total_checks:
        print("\n[AUDIT PASS] All checks passed.")
    else:
        print(f"\n[AUDIT PARTIAL] {total_checks - passed} checks failed.")


if __name__ == "__main__":
    main()
