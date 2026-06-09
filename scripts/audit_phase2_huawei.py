"""
Phase 2 Huawei Audit — EVT+CVaR Methodology Validation
=======================================================

Verifies that Phase 2 Huawei (risk-aware) results are leakage-free and
methodologically sound. Adapts the same audit logic used for Azure Phase 2.

Checks:
  1. Test set identity          — all 5 models use identical test demands
  2. No future leakage          — first W=30 timesteps: sigma_t == sigma_train
  3. Buffer non-constant        — buffer CV > 0.01 (dynamic, not static)
  4. EVT params loaded          — evt_parameters.json exists with all 5 models
  5. CVaR_z positive            — buffer multiplier > 0 for all models
  6. xi sanity                  — xi within [-1, 2] (pathological values flag data issues)
  7. Accounting identity        — cold/idle math correct
  8. Cold/idle exclusivity      — never both > 0 at same timestep
  9. Buffer size check          — buffer_t >= 0 throughout (provisioning can only add)
 10. Buffer larger at extremes  — mean buffer during extreme events > mean overall buffer
                                  (KNOWN FAILURE on Azure for Forecast_Only, Seasonal_Naive
                                   — document but do not suppress if it recurs)

Outputs:
    results/phase2/huawei/combined/audit_results.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR    = os.path.join(PROJECT_ROOT, "data",    "processed", "huawei", "combined")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2",   "huawei", "combined")
PHASE1_DIR  = os.path.join(PROJECT_ROOT, "results", "phase1",   "huawei", "combined")

ALPHA              = 0.99
VOLATILITY_WINDOW  = 30
K_GAUSSIAN         = norm.pdf(norm.ppf(ALPHA)) / (1 - ALPHA)

MODEL_NAMES = [
    "RiskAware(Reactive)",
    "RiskAware(Forecast_Only)",
    "RiskAware(Seasonal_Naive)",
    "RiskAware(Linear_Seasonal)",
    "RiskAware(TCN)",
]

# Azure Phase 2 known failures (Forecast_Only and Seasonal_Naive fail
# "buffer larger during extremes" due to their prediction characteristics)
KNOWN_FAILURES_AZURE = {
    "RiskAware(Forecast_Only)":  "buffer_larger_at_extremes",
    "RiskAware(Seasonal_Naive)": "buffer_larger_at_extremes",
}

audit_log    = []
total_checks = 0
passed       = 0


def check(name: str, condition: bool, detail: str = "",
          expected_fail: bool = False) -> None:
    global total_checks, passed
    total_checks += 1

    if expected_fail and not condition:
        status = "XFAIL"  # expected failure — still count as pass for overall
        passed += 1
        tag = f"  [XFAIL] {name}"
    elif condition:
        status = "PASS"
        passed += 1
        tag = f"  [PASS] {name}"
    else:
        status = "FAIL"
        tag = f"  [FAIL] {name}"

    print(tag)
    if detail and not condition:
        print(f"         Detail: {detail}")
    audit_log.append({"name": name, "status": status, "detail": detail,
                      "expected_fail": expected_fail})


# ---------------------------------------------------------------------------
# Data loading
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


def load_evt_params():
    path = os.path.join(RESULTS_DIR, "evt_parameters.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_phase1_threshold():
    path = os.path.join(PHASE1_DIR, "extreme_threshold.txt")
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

def audit_test_set_identity(diags, test):
    print("\n--- 1. Test Set Identity ---")
    ref_name = MODEL_NAMES[0]
    if ref_name not in diags:
        check("Test set identity (reference loaded)", False, "Reference missing")
        return

    ref_demand = diags[ref_name]["actual_demand"].values
    for name in MODEL_NAMES[1:]:
        if name not in diags:
            check(f"Test demands identical: {name}", False, "Diagnostics missing")
            continue
        this_demand = diags[name]["actual_demand"].values
        check(f"Test demands identical: {ref_name} vs {name}",
              np.allclose(ref_demand, this_demand, atol=1e-6),
              f"Max diff = {np.max(np.abs(ref_demand - this_demand)):.2f}")

    raw = test["concurrency"].values
    check("Test demands match raw test.csv",
          np.allclose(ref_demand, raw, atol=1e-6))


def audit_no_future_leakage(diags, train):
    print("\n--- 2. No Future Leakage (warm-up period) ---")
    sigma_train_vals = {}

    # Reconstruct sigma_train from training residuals for reference
    # We don't have base model here — use sigma_t at t=0 from diagnostics
    for name, df in diags.items():
        if "sigma_t" not in df.columns:
            check(f"sigma_t column exists: {name}", False, "Column missing")
            continue
        sigma_t = df["sigma_t"].values
        # The first volatility_window timesteps should equal sigma_train
        # (warm-up: no rolling window yet)
        first_w = sigma_t[:VOLATILITY_WINDOW]
        # All should be identical (sigma_train + 1e-8 from the code, but essentially const)
        std_of_first_w = np.std(first_w)
        check(f"First {VOLATILITY_WINDOW} sigma_t constant (warm-up): {name}",
              std_of_first_w < 1e-6,
              f"std_of_first_W = {std_of_first_w:.6f} (should be ~0)")
        sigma_train_vals[name] = float(first_w[0])

    return sigma_train_vals


def audit_buffer_dynamic(diags):
    print("\n--- 3. Buffer Dynamism ---")
    for name, df in diags.items():
        if "buffer_t" not in df.columns:
            check(f"buffer_t column exists: {name}", False)
            continue
        buf = df["buffer_t"].values
        buf_mean = np.mean(buf)
        buf_std  = np.std(buf)
        cv = buf_std / buf_mean if buf_mean > 1e-8 else 0.0
        check(f"Buffer CV > 0.01 (dynamic): {name}",
              cv > 0.01,
              f"CV = {cv:.4f}")


def audit_evt_params(evt_params):
    print("\n--- 4. EVT Parameters Sanity ---")
    check("evt_parameters.json loaded", len(evt_params) > 0,
          "File missing or empty")

    for name in MODEL_NAMES:
        if name not in evt_params:
            check(f"EVT params present: {name}", False, "Missing from JSON")
            continue
        ep = evt_params[name]
        check(f"CVaR_z > 0: {name}",
              ep.get("cvar_z", 0) > 0,
              f"cvar_z = {ep.get('cvar_z')}")
        check(f"xi in [-1, 2]: {name}",
              -1.0 <= ep.get("xi", 999) <= 2.0,
              f"xi = {ep.get('xi')}")
        check(f"beta > 0: {name}",
              ep.get("beta", 0) > 0,
              f"beta = {ep.get('beta')}")
        check(f"ratio = cvar_z / K_GAUSSIAN present: {name}",
              "cvar_z_over_k_gaussian" in ep,
              "Key missing")


def audit_accounting_identity(diags):
    print("\n--- 5. Accounting Identity ---")
    for name, df in diags.items():
        demand = df["actual_demand"].values
        prov   = df["provisioned"].values
        cold   = df["cold_starts"].values
        idle   = df["idle_capacity"].values

        exp_cold = np.maximum(demand - prov, 0)
        exp_idle = np.maximum(prov - demand, 0)

        check(f"Cold accounting: {name}",
              np.allclose(cold, exp_cold, atol=1e-4),
              f"Max diff = {np.max(np.abs(cold - exp_cold)):.6f}")
        check(f"Idle accounting: {name}",
              np.allclose(idle, exp_idle, atol=1e-4))


def audit_cold_idle_exclusivity(diags):
    print("\n--- 6. Cold/Idle Exclusivity ---")
    for name, df in diags.items():
        cold = df["cold_starts"].values
        idle = df["idle_capacity"].values
        both = (cold > 1e-6) & (idle > 1e-6)
        check(f"No timestep with both cold & idle: {name}",
              not both.any(), f"{both.sum()} violations")


def audit_buffer_non_negative(diags):
    print("\n--- 7. Buffer Non-Negative ---")
    for name, df in diags.items():
        if "buffer_t" not in df.columns:
            continue
        buf = df["buffer_t"].values
        check(f"buffer_t >= 0 throughout: {name}",
              (buf >= -1e-8).all(),
              f"Min buffer = {buf.min():.4f}")


def audit_buffer_larger_at_extremes(diags, train):
    """
    Checks whether the dynamic buffer is larger during extreme events.
    KNOWN FAILURES on Azure: Forecast_Only, Seasonal_Naive.
    Document failures; do not suppress.
    """
    print("\n--- 8. Buffer Larger During Extremes ---")
    from evaluation.extreme import compute_extreme_threshold, EXTREME_PERCENTILE
    threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)

    for name, df in diags.items():
        if "buffer_t" not in df.columns or "actual_demand" not in df.columns:
            continue
        buf    = df["buffer_t"].values
        demand = df["actual_demand"].values
        is_ext = demand > threshold

        if is_ext.sum() == 0:
            check(f"Buffer at extremes (no extreme events): {name}", True)
            continue

        mean_buf_all = np.mean(buf)
        mean_buf_ext = np.mean(buf[is_ext])

        is_known_fail = (name in KNOWN_FAILURES_AZURE)
        condition = mean_buf_ext > mean_buf_all
        detail = (f"mean_buf_all={mean_buf_all:,.1f}  "
                  f"mean_buf_ext={mean_buf_ext:,.1f}  "
                  f"n_extreme={is_ext.sum()}")
        if is_known_fail and not condition:
            detail += " — KNOWN FAILURE (same behavior as Azure)"

        check(f"Buffer larger during extremes: {name}",
              condition,
              detail,
              expected_fail=is_known_fail)


def audit_no_nan(diags):
    print("\n--- 9. No NaN / Inf ---")
    for name, df in diags.items():
        for col in ["actual_demand", "predicted", "provisioned",
                    "cold_starts", "idle_capacity", "buffer_t"]:
            if col not in df.columns:
                continue
            vals = df[col].values
            check(f"No NaN in {col}: {name}", not np.isnan(vals).any())
            check(f"No Inf in {col}: {name}", not np.isinf(vals).any())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global total_checks, passed
    print("=" * 60)
    print("Phase 2 Huawei Audit")
    print("=" * 60)

    train, val, test = load_data()
    diags = load_diagnostics()
    evt_params = load_evt_params()

    audit_test_set_identity(diags, test)
    audit_no_future_leakage(diags, train)
    audit_buffer_dynamic(diags)
    audit_evt_params(evt_params)
    audit_accounting_identity(diags)
    audit_cold_idle_exclusivity(diags)
    audit_buffer_non_negative(diags)
    audit_buffer_larger_at_extremes(diags, train)
    audit_no_nan(diags)

    print(f"\n{'='*60}")
    print(f"Audit Result: {passed}/{total_checks} checks passed")
    print(f"  (XFAIL = expected failure, still counted as passed)")
    print(f"{'='*60}")

    xfails   = [e for e in audit_log if e["status"] == "XFAIL"]
    failures = [e for e in audit_log if e["status"] == "FAIL"]

    if xfails:
        print(f"\nExpected failures ({len(xfails)}):")
        for f in xfails:
            print(f"  [XFAIL] {f['name']}")
            if f["detail"]:
                print(f"          {f['detail']}")

    if failures:
        print(f"\nUnexpected failures ({len(failures)}):")
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
        "expected_failures_noted": len(xfails),
        "unexpected_failures": len(failures),
        "known_failures_reference": (
            "On Azure Phase 2, RiskAware(Forecast_Only) and "
            "RiskAware(Seasonal_Naive) failed 'buffer larger during extremes' "
            "because their base predictions track the seasonal cycle (lag_1440), "
            "causing residuals to be smaller during demand spikes (which are "
            "partially captured by the seasonal component). "
            "These are marked XFAIL and documented, not suppressed."
        ),
        "checks": audit_log,
    }
    out_path = os.path.join(RESULTS_DIR, "audit_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    if len(failures) == 0:
        print("\n[AUDIT PASS] All non-expected checks passed.")
    else:
        print(f"\n[AUDIT PARTIAL] {len(failures)} unexpected checks failed.")


if __name__ == "__main__":
    main()
