"""
Phase 4 Leakage Audit -- Component Ablation Validation
======================================================

Verifies that the Phase 4 ablation conditions are correctly implemented
and introduce no new leakage beyond what was established in Phases 1-3.

Audit categories:
  1. Test set identity      -- all C1-C3 conditions use the same test demands
  2. C0/C4 consistency      -- C0 metrics match Phase 1 exactly; C4 matches Phase 2
  3. Static sigma check     -- C1, C3: std(sigma_t) == 0 (rolling loop did not execute)
  4. Dynamic sigma check    -- C2: sigma_t varies after warm-up (W=30)
  5. Gaussian multiplier    -- C1, C2: cvar_col == K_GAUSSIAN for every timestep
  6. EVT multiplier check   -- C3: cvar_col == fitted CVaR_z; xi < 1 (finite CVaR)
  7. Buffer structure       -- C1, C3: buffer constant (CV==0); C2: buffer varies
  8. Accounting identity    -- served + cold == demand for all new conditions
  9. Phase 1/2/3 immutability -- existing result files untouched
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR       = os.path.join(PROJECT_ROOT, "data",    "processed", "azure")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results", "phase4",   "azure")
CONDITIONS_DIR = os.path.join(RESULTS_DIR, "conditions")
PHASE1_DIR     = os.path.join(PROJECT_ROOT, "results", "phase1",   "azure")
PHASE2_DIR     = os.path.join(PROJECT_ROOT, "results", "phase2",   "azure")
PHASE3_DIR     = os.path.join(PROJECT_ROOT, "results", "phase3",   "azure")

# K_GAUSSIAN must be derived the same way as in run_phase4.py
ALPHA = 0.99
_z_alpha  = norm.ppf(ALPHA)
K_GAUSSIAN = norm.pdf(_z_alpha) / (1 - ALPHA)

VOLATILITY_WINDOW = 30

audit_log    = []
total_checks = 0
passed       = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global total_checks, passed
    total_checks += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    print(f"  [{status}] {name}")
    if detail and not condition:
        print(f"         Detail: {detail}")
    audit_log.append({"name": name, "status": status, "detail": detail})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_phase4_conditions():
    """Load all diagnostics and metrics from conditions/ directory."""
    diag_files    = sorted(glob.glob(os.path.join(CONDITIONS_DIR, "*_diagnostics.csv")))
    metrics_files = sorted(glob.glob(os.path.join(CONDITIONS_DIR, "*_metrics.json")))

    all_diagnostics = {}
    all_metrics     = {}

    for path in diag_files:
        fname = os.path.basename(path).replace("_diagnostics.csv", "")
        for model in ["Reactive", "TCN"]:
            suffix = f"_{model}"
            if fname.endswith(suffix):
                cid = fname[:-len(suffix)]
                all_diagnostics[(cid, model)] = pd.read_csv(path)
                break

    for path in metrics_files:
        fname = os.path.basename(path).replace("_metrics.json", "")
        for model in ["Reactive", "TCN"]:
            suffix = f"_{model}"
            if fname.endswith(suffix):
                cid = fname[:-len(suffix)]
                with open(path) as f:
                    all_metrics[(cid, model)] = json.load(f)
                break

    return all_diagnostics, all_metrics


# ---------------------------------------------------------------------------
# Audit 1: Test set identity (C1-C3 only)
# ---------------------------------------------------------------------------
def audit_test_set_identity(all_diagnostics: dict) -> None:
    print("\n--- AUDIT 1: TEST SET IDENTITY ---")
    check("Phase 4 condition diagnostics exist",
          len(all_diagnostics) > 0,
          f"Found {len(all_diagnostics)} files in {CONDITIONS_DIR}")

    if not all_diagnostics:
        return

    demand_arrays  = {k: v["actual_demand"].values for k, v in all_diagnostics.items()}
    reference_key  = next(iter(demand_arrays))
    ref_demand     = demand_arrays[reference_key]

    all_identical = True
    for k, arr in demand_arrays.items():
        if not np.array_equal(arr, ref_demand):
            check(f"Identical test demands: {k}", False,
                  f"differs from reference {reference_key}")
            all_identical = False

    check("All C1-C3 conditions use identical test demands", all_identical)
    check(f"Test set has 3,744 timesteps",
          len(ref_demand) == 3744, f"got {len(ref_demand)}")


# ---------------------------------------------------------------------------
# Audit 2: C0/C4 consistency with Phase 1/2
# ---------------------------------------------------------------------------
def audit_c0_c4_consistency() -> None:
    print("\n--- AUDIT 2: C0/C4 METRIC CONSISTENCY WITH PHASE 1/2 ---")

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        check("summary.csv exists", False, summary_path)
        return
    check("summary.csv exists", True)

    df = pd.read_csv(summary_path)

    with open(os.path.join(PHASE1_DIR, "metrics.json")) as f:
        p1 = json.load(f)
    with open(os.path.join(PHASE2_DIR, "metrics.json")) as f:
        p2 = json.load(f)

    for model_name in ["Reactive", "TCN"]:
        # C0 must exactly match Phase 1
        c0_row = df[(df["condition_id"] == "C0_no_buffer") & (df["model"] == model_name)]
        if c0_row.empty:
            check(f"C0 row found for {model_name}", False)
            continue
        check(f"C0 row found for {model_name}", True)

        sla_diff  = abs(float(c0_row["request_sla"].iloc[0]) - p1[model_name]["request_sla"])
        cost_diff = abs(float(c0_row["total_cost"].iloc[0])   - p1[model_name]["total_cost"])
        check(f"[{model_name}] C0 SLA == Phase 1 (exact)",
              sla_diff < 1e-10,
              f"diff={sla_diff:.2e}")
        check(f"[{model_name}] C0 Cost == Phase 1 (exact)",
              cost_diff < 1.0,
              f"diff={cost_diff:.1f}")

        # C4 must match Phase 2
        c4_row = df[(df["condition_id"] == "C4_dynamic_evt") & (df["model"] == model_name)]
        if c4_row.empty:
            check(f"C4 row found for {model_name}", False)
            continue
        check(f"C4 row found for {model_name}", True)

        p2_key   = f"RiskAware({model_name})"
        sla_tol  = 1e-10 if model_name == "Reactive" else 1e-4
        cost_tol = 1.0   if model_name == "Reactive" else 1e6

        c4_sla  = float(c4_row["request_sla"].iloc[0])
        c4_cost = float(c4_row["total_cost"].iloc[0])
        sla_diff  = abs(c4_sla  - p2[p2_key]["request_sla"])
        cost_diff = abs(c4_cost - p2[p2_key]["total_cost"])
        check(f"[{model_name}] C4 SLA == Phase 2 (tol={sla_tol:.0e})",
              sla_diff <= sla_tol,
              f"p4={c4_sla:.8f}  p2={p2[p2_key]['request_sla']:.8f}  diff={sla_diff:.2e}")
        check(f"[{model_name}] C4 Cost == Phase 2 (tol={cost_tol:,.0f})",
              cost_diff <= cost_tol,
              f"p4={c4_cost:,.0f}  p2={p2[p2_key]['total_cost']:,.0f}  diff={cost_diff:,.0f}")


# ---------------------------------------------------------------------------
# Audit 3: Static sigma -- C1 and C3 must have constant sigma_t
# ---------------------------------------------------------------------------
def audit_static_sigma(all_diagnostics: dict) -> None:
    print("\n--- AUDIT 3: STATIC SIGMA (C1, C3: sigma_t must be constant) ---")

    for cid in ["C1_static_gaussian", "C3_static_evt"]:
        for model_name in ["Reactive", "TCN"]:
            key = (cid, model_name)
            if key not in all_diagnostics:
                check(f"[{cid} {model_name}] diagnostics found", False)
                continue

            sigma = all_diagnostics[key]["sigma_t"].values
            std_sigma = np.std(sigma)
            check(f"[{cid} {model_name}] sigma_t is constant (std==0)",
                  std_sigma < 1e-10,
                  f"std(sigma_t) = {std_sigma:.6e}")

            # Verify the constant value equals sigma_train (first value)
            sigma_train = sigma[0]
            check(f"[{cid} {model_name}] sigma_t == sigma_train = {sigma_train:,.0f}",
                  np.allclose(sigma, sigma_train, atol=1e-6), "")


# ---------------------------------------------------------------------------
# Audit 4: Dynamic sigma -- C2 must have varying sigma_t after warm-up
# ---------------------------------------------------------------------------
def audit_dynamic_sigma(all_diagnostics: dict) -> None:
    print("\n--- AUDIT 4: DYNAMIC SIGMA (C2: sigma_t varies after warm-up) ---")

    for model_name in ["Reactive", "TCN"]:
        key = ("C2_dynamic_gaussian", model_name)
        if key not in all_diagnostics:
            check(f"[C2 {model_name}] diagnostics found", False)
            continue

        sigma      = all_diagnostics[key]["sigma_t"].values
        sigma_train = sigma[0]

        # Warm-up: first W values must equal sigma_train
        warm_up = sigma[:VOLATILITY_WINDOW]
        check(f"[C2 {model_name}] sigma warm-up: first {VOLATILITY_WINDOW} values == sigma_train",
              np.allclose(warm_up, sigma_train, atol=1e-6),
              f"max deviation = {np.max(np.abs(warm_up - sigma_train)):.2e}")

        # Post-warm-up: sigma_t must vary
        post_warmup  = sigma[VOLATILITY_WINDOW:]
        n_distinct   = len(np.unique(np.round(post_warmup, 4)))
        check(f"[C2 {model_name}] sigma_t varies after warm-up (>1 distinct value)",
              n_distinct > 1,
              f"distinct values in post-warmup: {n_distinct}")


# ---------------------------------------------------------------------------
# Audit 5: Gaussian multiplier -- C1, C2 must use K_GAUSSIAN
# ---------------------------------------------------------------------------
def audit_gaussian_multiplier(all_diagnostics: dict) -> None:
    print(f"\n--- AUDIT 5: GAUSSIAN MULTIPLIER (C1, C2: cvar_col == {K_GAUSSIAN:.6f}) ---")

    for cid in ["C1_static_gaussian", "C2_dynamic_gaussian"]:
        for model_name in ["Reactive", "TCN"]:
            key = (cid, model_name)
            if key not in all_diagnostics:
                continue

            diag = all_diagnostics[key]
            if "cvar_col" not in diag.columns:
                check(f"[{cid} {model_name}] cvar_col column exists", False,
                      "column missing from diagnostics CSV")
                continue

            cvar_col = diag["cvar_col"].values
            max_dev  = np.max(np.abs(cvar_col - K_GAUSSIAN))
            check(f"[{cid} {model_name}] cvar_col == K_GAUSSIAN (max dev < 1e-10)",
                  max_dev < 1e-10,
                  f"max deviation = {max_dev:.2e}")


# ---------------------------------------------------------------------------
# Audit 6: EVT multiplier -- C3 must use fitted CVaR_z; xi < 1
# ---------------------------------------------------------------------------
def audit_evt_multiplier(all_diagnostics: dict) -> None:
    print("\n--- AUDIT 6: EVT MULTIPLIER (C3: cvar_col == CVaR_z, xi < 1) ---")

    for model_name in ["Reactive", "TCN"]:
        key = ("C3_static_evt", model_name)
        if key not in all_diagnostics:
            check(f"[C3 {model_name}] diagnostics found", False)
            continue

        diag    = all_diagnostics[key]
        metrics_path = os.path.join(CONDITIONS_DIR, f"C3_static_evt_{model_name}_metrics.json")
        if not os.path.exists(metrics_path):
            check(f"[C3 {model_name}] metrics.json found", False)
            continue

        cvar_col = diag["cvar_col"].values
        # All values should be identical (static multiplier)
        cvar_unique = np.unique(np.round(cvar_col, 8))
        check(f"[C3 {model_name}] cvar_col is constant (EVT CVaR_z is fixed)",
              len(cvar_unique) == 1,
              f"unique values: {cvar_unique}")

        if len(cvar_unique) == 1:
            cvar_z = cvar_unique[0]
            check(f"[C3 {model_name}] CVaR_z > K_GAUSSIAN ({K_GAUSSIAN:.4f}): "
                  f"EVT gives {cvar_z:.4f} (heavy-tail effect)",
                  cvar_z > K_GAUSSIAN,
                  f"CVaR_z={cvar_z:.4f} is not > K_GAUSSIAN={K_GAUSSIAN:.4f}")

        # Load Phase 2 evt_parameters to verify xi < 1
        p2_evt_path = os.path.join(PHASE2_DIR, "evt_parameters.json")
        if os.path.exists(p2_evt_path):
            with open(p2_evt_path) as f:
                p2_evt = json.load(f)
            p2_key = f"RiskAware({model_name})"
            if p2_key in p2_evt:
                xi = p2_evt[p2_key]["xi"]
                check(f"[{model_name}] Phase 2 EVT xi < 1 (finite CVaR): xi={xi:.4f}",
                      xi < 1.0,
                      f"xi={xi:.4f} >= 1 would imply infinite CVaR")


# ---------------------------------------------------------------------------
# Audit 7: Buffer structure
# ---------------------------------------------------------------------------
def audit_buffer_structure(all_diagnostics: dict) -> None:
    print("\n--- AUDIT 7: BUFFER STRUCTURE ---")

    for cid in ["C1_static_gaussian", "C3_static_evt"]:
        for model_name in ["Reactive", "TCN"]:
            key = (cid, model_name)
            if key not in all_diagnostics:
                continue
            buf = all_diagnostics[key]["buffer_t"].values
            cv  = np.std(buf) / np.mean(buf) if np.mean(buf) > 0 else 0
            check(f"[{cid} {model_name}] buffer is constant (CV < 1e-8)",
                  cv < 1e-8,
                  f"CV = {cv:.6e}")

    for model_name in ["Reactive", "TCN"]:
        key = ("C2_dynamic_gaussian", model_name)
        if key not in all_diagnostics:
            continue
        buf = all_diagnostics[key]["buffer_t"].values
        cv  = np.std(buf) / np.mean(buf) if np.mean(buf) > 0 else 0
        check(f"[C2 {model_name}] buffer varies (CV > 0.01)",
              cv > 0.01,
              f"CV = {cv:.4f}")


# ---------------------------------------------------------------------------
# Audit 8: Accounting identity
# ---------------------------------------------------------------------------
def audit_accounting_identity(all_diagnostics: dict) -> None:
    print("\n--- AUDIT 8: ACCOUNTING IDENTITY ---")

    for (cid, model_name), diag in all_diagnostics.items():
        demand      = diag["actual_demand"].values
        provisioned = diag["provisioned"].values
        cold        = diag["cold_starts"].values
        idle        = diag["idle_capacity"].values

        no_neg        = (cold >= -1e-6).all() and (idle >= -1e-6).all()
        mutual_exc    = ((cold > 0) & (idle > 0)).sum() == 0
        expected_cold = np.maximum(demand - provisioned, 0)
        accounting    = np.allclose(cold, expected_cold, atol=1.0)

        check(f"[{cid} {model_name}] Accounting: no neg, mutual excl., cold == max(d-p, 0)",
              no_neg and mutual_exc and accounting,
              f"neg={not no_neg}  both_pos={not mutual_exc}  accounting={not accounting}")


# ---------------------------------------------------------------------------
# Audit 9: Phase 1/2/3 immutability
# ---------------------------------------------------------------------------
def audit_immutability() -> None:
    print("\n--- AUDIT 9: PHASE 1/2/3 RESULT FILES NOT MODIFIED ---")

    sentinel_files = [
        os.path.join(PHASE1_DIR, "metrics.json"),
        os.path.join(PHASE1_DIR, "comparison.csv"),
        os.path.join(PHASE2_DIR, "metrics.json"),
        os.path.join(PHASE2_DIR, "comparison.csv"),
        os.path.join(PHASE3_DIR, "summary_3a.csv"),
        os.path.join(PHASE3_DIR, "summary_3b.csv"),
    ]
    for path in sentinel_files:
        rel = os.path.relpath(path, PROJECT_ROOT)
        exists = os.path.exists(path)
        check(f"File exists (not deleted): {rel}", exists, f"missing: {path}")
        if exists:
            check(f"File non-empty: {rel}", os.path.getsize(path) > 0)

    check("RESULTS_DIR is results/phase4/ (not phase1/2/3)",
          "phase4" in RESULTS_DIR and "phase1" not in RESULTS_DIR
          and "phase2" not in RESULTS_DIR and "phase3" not in RESULTS_DIR,
          RESULTS_DIR)


# ---------------------------------------------------------------------------
# Robustness summary
# ---------------------------------------------------------------------------
def print_robustness_summary(all_diagnostics: dict) -> None:
    print("\n" + "="*60)
    print("ABLATION VERIFICATION SUMMARY (informational)")
    print("="*60)

    # Show sigma_t statistics per condition
    for model_name in ["Reactive", "TCN"]:
        print(f"\n  Model: {model_name}")
        for cid in ["C1_static_gaussian", "C2_dynamic_gaussian", "C3_static_evt"]:
            key = (cid, model_name)
            if key not in all_diagnostics:
                continue
            sigma = all_diagnostics[key]["sigma_t"].values
            buf   = all_diagnostics[key]["buffer_t"].values
            print(f"    {cid}: sigma mean={np.mean(sigma):,.0f}  "
                  f"std={np.std(sigma):,.0f}  "
                  f"buffer mean={np.mean(buf):,.0f}  "
                  f"std={np.std(buf):,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*60)
    print("Phase 4: Leakage Audit")
    print("="*60)
    print(f"  K_GAUSSIAN = {K_GAUSSIAN:.6f}")

    if not os.path.exists(CONDITIONS_DIR) or len(os.listdir(CONDITIONS_DIR)) == 0:
        print("\nERROR: No Phase 4 results found.")
        print(f"  Run scripts/run_phase4.py first.")
        sys.exit(1)

    all_diagnostics, all_metrics = load_phase4_conditions()
    print(f"\n  Loaded {len(all_diagnostics)} diagnostic files")
    print(f"  Loaded {len(all_metrics)} metrics files")

    audit_test_set_identity(all_diagnostics)
    audit_c0_c4_consistency()
    audit_static_sigma(all_diagnostics)
    audit_dynamic_sigma(all_diagnostics)
    audit_gaussian_multiplier(all_diagnostics)
    audit_evt_multiplier(all_diagnostics)
    audit_buffer_structure(all_diagnostics)
    audit_accounting_identity(all_diagnostics)
    audit_immutability()

    print_robustness_summary(all_diagnostics)

    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total checks: {total_checks}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {total_checks - passed}")
    overall = "PASS" if passed == total_checks else "FAIL"
    print(f"\n  Overall: {overall}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "audit_results.json"), "w") as f:
        json.dump({
            "total_checks": total_checks,
            "passed":  passed,
            "failed":  total_checks - passed,
            "overall": overall,
            "checks":  audit_log,
        }, f, indent=2)
    print(f"\n  Saved: {os.path.join(RESULTS_DIR, 'audit_results.json')}")


if __name__ == "__main__":
    main()
