"""
Phase 2: Risk-Aware Model — Training, Evaluation & Comparison
==============================================================
Trains the EVT+CVaR risk-aware scaler and compares it against
all Phase 1 baselines on the same test set.

Usage:
    python scripts/run_risk_aware.py

Output:
    - Component-level sanity checks
    - Fair comparison table (all models, same test set, same metrics)
    - Extreme event analysis
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import create_baselines, compare_models, print_comparison, train_mlp_scaler
from evaluation.config import COST_COLD, COST_IDLE
from models.risk_aware import (
    train_quantile_models,
    fit_evt_model,
    generate_scenarios,
    optimize_cvar,
    RiskAwareScaler,
)


def load_data(dataset_subdir):
    """Load train and test data for a dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'dataset', 'processed', dataset_subdir, 'train.csv')
    test_path = os.path.join(base_dir, 'dataset', 'processed', dataset_subdir, 'test.csv')

    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])

    return train_data, test_data


def sanity_check_components(quantile_models, evt_params, test_data):
    """
    Print component-level diagnostics for human verification.
    """
    print("\n" + "=" * 70)
    print("COMPONENT SANITY CHECKS")
    print("=" * 70)

    # --- Quantile model: check 5 random test points ---
    print("\n[1] Quantile predictions (5 random test points):")
    feature_cols = [f'lag_{k}' for k in range(1, 11)]
    np.random.seed(42)
    sample_idx = np.random.choice(len(test_data), 5, replace=False)

    for idx in sorted(sample_idx):
        features = test_data.iloc[idx][feature_cols].values.reshape(1, -1)
        actual = test_data.iloc[idx]['concurrency']
        q50 = quantile_models[0.50].predict(features)[0]
        q90 = quantile_models[0.90].predict(features)[0]
        q99 = quantile_models[0.99].predict(features)[0]
        ordered = "OK" if q50 <= q90 <= q99 else "X CROSSING"
        print(f"    t={idx:4d}: actual={actual:>10,.0f}  "
              f"q50={q50:>10,.0f}  q90={q90:>10,.0f}  q99={q99:>10,.0f}  {ordered}")

    # --- EVT: already printed during fit ---
    print(f"\n[2] EVT params: threshold={evt_params['threshold']:,.0f}, "
          f"shape={evt_params['shape']:.4f}, scale={evt_params['scale']:,.2f}")

    # --- Scenario generation: one sample ---
    print("\n[3] Scenario generation (one test point):")
    features = test_data.iloc[0][feature_cols].values.reshape(1, -1)
    q50 = quantile_models[0.50].predict(features)[0]
    q90 = max(quantile_models[0.90].predict(features)[0], q50)
    q99 = max(quantile_models[0.99].predict(features)[0], q90)
    scenarios = generate_scenarios(q50, q90, q99, evt_params, n_scenarios=300)
    print(f"    q50={q50:,.0f}, q90={q90:,.0f}, q99={q99:,.0f}")
    print(f"    Scenarios: min={scenarios.min():,.0f}, mean={scenarios.mean():,.0f}, "
          f"max={scenarios.max():,.0f}, std={scenarios.std():,.0f}")

    # --- CVaR optimization: one sample ---
    print("\n[4] CVaR optimization (one test point):")
    c_star = optimize_cvar(scenarios, alpha=0.05,
                           cost_cold=COST_COLD, cost_idle=COST_IDLE)
    actual = test_data.iloc[0]['concurrency']
    print(f"    c*={c_star:,.0f} (actual demand={actual:,.0f}, q90={q90:,.0f}, q99={q99:,.0f})")
    print(f"    c* vs q90: {'+' if c_star > q90 else ''}{c_star - q90:,.0f}")


def run_dataset(dataset_subdir, dataset_name):
    """Train and evaluate on one dataset."""
    print("\n" + "=" * 90)
    print(f"  DATASET: {dataset_name}")
    print("=" * 90)

    # Load data
    print(f"\n[SETUP] Loading {dataset_name} data...")
    train_data, test_data = load_data(dataset_subdir)
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Test:  {len(test_data):,} samples")

    # ====================================================================
    # PHASE 1: Create baselines (same as before)
    # ====================================================================

    print("\n[BASELINES] Creating Phase 1 models...")
    baselines = create_baselines(train_data)
    print(f"  Created: {', '.join(baselines.keys())}")

    print("  Training MLP baseline...", end=" ", flush=True)
    mlp = train_mlp_scaler(train_data, verbose=False)
    baselines['MLP Forecast'] = mlp
    print("done")

    # ====================================================================
    # PHASE 2: Train risk-aware model
    # ====================================================================

    print(f"\n[RISK-AWARE] Training EVT+CVaR model...")

    # Component 1: Quantile models
    print("\n  --- Quantile Forecaster ---")
    q_models = train_quantile_models(train_data, verbose=True)

    # Component 2: EVT
    print("\n  --- EVT Tail Model ---")
    evt_params = fit_evt_model(train_data['concurrency'].values, verbose=True)

    # Component 5: Assemble wrapper
    risk_aware = RiskAwareScaler(
        quantile_models=q_models,
        evt_params=evt_params,
        alpha=0.05,
        n_scenarios=300,
        cost_cold=COST_COLD,
        cost_idle=COST_IDLE,
    )

    # Sanity checks
    sanity_check_components(q_models, evt_params, test_data)

    # ====================================================================
    # FAIR COMPARISON: All models on same test set
    # ====================================================================

    all_models = {**baselines, 'EVT+CVaR (Ours)': risk_aware}

    print(f"\n[EVAL] Running fair comparison ({len(all_models)} models)...")
    print(f"  Cost params: C_cold={COST_COLD}, C_idle={COST_IDLE}")

    comparison_df, detailed = compare_models(
        models_dict=all_models,
        test_data=test_data,
        dataset_name=dataset_name,
        cost_cold=COST_COLD,
        cost_idle=COST_IDLE,
        percentile_extreme=99,
        verbose=False,
    )

    # ====================================================================
    # PRINT RESULTS
    # ====================================================================

    print("\n" + "=" * 90)
    print(f"  RESULTS: {dataset_name}")
    print("=" * 90)

    print("\n" + comparison_df.to_string())

    # Ranking
    print("\n\nRANKING BY TOTAL COST:")
    costs = [(idx, comparison_df.loc[idx, 'Total Cost']) for idx in comparison_df.index]
    costs.sort(key=lambda x: x[1])
    for rank, (name, cost) in enumerate(costs, 1):
        extreme_sla = detailed[name]['extreme_metrics'].get('sla_on_extreme', 0)
        marker = " << OURS" if name == 'EVT+CVaR (Ours)' else ""
        print(f"  {rank}. {name:25s} | Cost: {cost:>13,.0f} "
              f"| Extreme SLA: {extreme_sla:>6.1%}{marker}")

    # Highlight improvement
    print("\n\nEXTREME EVENT COMPARISON:")
    for name in all_models:
        ext = detailed[name]['extreme_metrics']
        summ = detailed[name]['summary']
        print(f"  {name:25s} | Overall SLA: {summ['sla_compliance']:>6.1%} "
              f"| Extreme SLA: {ext.get('sla_on_extreme', 0):>6.1%} "
              f"| Avg Cold (extreme): {ext.get('cold_mean_on_extreme', 0):>10,.0f}")

    return comparison_df, detailed


def main():
    print("\n" + "=" * 90)
    print("  PHASE 2: RISK-AWARE AUTOSCALING (EVT + CVaR)")
    print("=" * 90)

    # Run on Azure dataset
    comparison_df, detailed = run_dataset('azure', 'Azure Functions')

    print("\n" + "=" * 90)
    print("  [OK] PHASE 2 EVALUATION COMPLETE")
    print("=" * 90 + "\n")

    return comparison_df, detailed


if __name__ == '__main__':
    comparison_df, detailed = main()
