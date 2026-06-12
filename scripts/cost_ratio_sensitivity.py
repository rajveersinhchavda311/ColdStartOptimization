"""
Cost-Ratio Sensitivity Analysis (read-only re-aggregation)
===========================================================

Question answered (reviewer-facing): "Your 10:1 cold:idle cost ratio is an
assumption — do the cost conclusions survive other ratios?"

Why this is EXACT, not approximate:
    Provisioning is ceil(prediction) and predictions never depend on the
    cost parameters — the cost model is evaluation-only. Per-timestep
    cold_starts and idle_capacity in the stored diagnostics are therefore
    invariant under any ratio, and

        total_cost(r) = r * total_cold_units + total_idle_units

    is a pure re-weighting of frozen per-timestep outcomes. No simulation,
    no training, no randomness.

Self-check:
    At r = 10 the recomputed cost must equal the stored total_cost in
    metrics.json EXACTLY (atol 1e-6). Any mismatch aborts the script.

Scope:
    Phase 1 Azure (6 models), Phase 2 Azure (5), Phase 1 Huawei combined (6),
    Phase 2 Huawei combined (5), Phase 4 Azure (C0-C4 x Reactive/TCN).

Output (additive only):
    results/analysis/cost_ratio_sensitivity.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RATIOS = [5, 10, 20]   # c_cold values; c_idle = 1 throughout
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")


def gather_units():
    """Collect (group, model, cold_units, idle_units, stored_total_cost) rows."""
    rows = []

    def add(group, model, metrics):
        # cold_cost = cold_units * 10, idle_cost = idle_units * 1
        rows.append({
            "group": group,
            "model": model,
            "cold_units": metrics["cold_cost"] / 10.0,
            "idle_units": metrics["idle_cost"] / 1.0,
            "stored_total_cost": metrics["total_cost"],
        })

    for group, path in [
        ("phase1_azure", "results/phase1/azure/metrics.json"),
        ("phase2_azure", "results/phase2/azure/metrics.json"),
        ("phase1_huawei", "results/phase1/huawei/combined/metrics.json"),
        ("phase2_huawei", "results/phase2/huawei/combined/metrics.json"),
    ]:
        with open(os.path.join(PROJECT_ROOT, path)) as f:
            for model, m in json.load(f).items():
                add(group, model, m)

    with open(os.path.join(PROJECT_ROOT, "results/phase4/azure/all_metrics.json")) as f:
        for cond, m in json.load(f).items():
            add("phase4_azure", cond, m)

    return pd.DataFrame(rows)


def main():
    df = gather_units()

    # Self-check at the baseline ratio: must reproduce stored totals exactly
    recomputed_10 = 10 * df["cold_units"] + df["idle_units"]
    deltas = (recomputed_10 - df["stored_total_cost"]).abs()
    if not (deltas < 1e-6).all():
        bad = df.loc[deltas >= 1e-6, ["group", "model"]]
        raise SystemExit(f"SELF-CHECK FAILED at r=10 — aborting. Mismatches:\n{bad}")
    print(f"Self-check PASS: r=10 reproduces all {len(df)} stored total_cost values exactly.")

    for r in RATIOS:
        df[f"cost_r{r}"] = r * df["cold_units"] + df["idle_units"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "cost_ratio_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows).")

    # Ranking stability report
    print("\n=== Cheapest model per group, per ratio ===")
    for group in df["group"].unique():
        sub = df[df["group"] == group]
        line = [group.ljust(15)]
        for r in RATIOS:
            best = sub.loc[sub[f"cost_r{r}"].idxmin(), "model"]
            line.append(f"r={r}: {best}")
        print("  " + " | ".join(line))

    print("\n=== Full cost tables (millions) ===")
    pd.set_option("display.width", 160)
    for group in df["group"].unique():
        sub = df[df["group"] == group].copy()
        for r in RATIOS:
            sub[f"cost_r{r}"] = (sub[f"cost_r{r}"] / 1e6).round(1)
        print(f"\n{group}:")
        print(sub[["model"] + [f"cost_r{r}" for r in RATIOS]].to_string(index=False))


if __name__ == "__main__":
    main()
