"""
Phase 1 Finalization: Complete baseline evaluation, graph generation, and results archiving.

This script performs the complete Phase 1 analysis:
1. Run baseline comparison on Azure dataset
2. Save results and intermediate data
3. Generate publication-ready graphs
4. Archive all outputs

Usage:
    python scripts/finalize_phase1.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (
    create_baselines, train_mlp_scaler, compare_models,
    identify_extreme_events, analyze_extreme_events
)
from evaluation.config import COST_COLD, COST_IDLE

# Output directories
RESULTS_DIR = 'results/phase1'
GRAPHS_DIR = 'graphs/phase1'
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
INTERMEDIATE_DIR = os.path.join(RESULTS_DIR, 'intermediate')

# Ensure directories exist
for dir_path in [TABLES_DIR, LOGS_DIR, INTERMEDIATE_DIR,
                 os.path.join(GRAPHS_DIR, 'comparison'),
                 os.path.join(GRAPHS_DIR, 'distribution'),
                 os.path.join(GRAPHS_DIR, 'timeseries')]:
    os.makedirs(dir_path, exist_ok=True)


def load_and_prepare_data():
    """Load train/test data and return processed DataFrames."""
    print("\n[PHASE 1] Loading datasets...")
    
    train_path = 'data/processed/azure/train.csv'
    test_path = 'data/processed/azure/test.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"ERROR: Data files not found")
        print(f"  Train: {train_path}")
        print(f"  Test:  {test_path}")
        return None, None
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    
    print(f"  [OK] Train data: {len(train_data)} samples")
    print(f"  [OK] Test data:  {len(test_data)} samples")
    
    return train_data, test_data


def run_baseline_comparison(train_data, test_data):
    """Run fair comparison of all baseline models."""
    print("\n[PHASE 1] Running baseline comparison...")
    
    # Create baselines
    print("  Creating baselines...")
    baselines = create_baselines(train_data)
    
    # Train MLP
    print("  Training MLP forecast...")
    mlp_scaler = train_mlp_scaler(train_data, verbose=False)
    baselines['MLP Forecast'] = mlp_scaler
    
    # Run comparison
    print("  Running fair comparison (4 models, 4030 timesteps)...")
    comparison_df, detailed_results = compare_models(
        models_dict=baselines,
        test_data=test_data,
        dataset_name='Azure Functions (Test Set)',
        cost_cold=COST_COLD,
        cost_idle=COST_IDLE,
        percentile_extreme=99,
        verbose=False,
    )
    
    return comparison_df, detailed_results, baselines, test_data


def save_results_and_intermediate(comparison_df, detailed_results, test_data, baselines):
    """Save comparison results and intermediate data for graphs."""
    print("\n[PHASE 1] Saving results and intermediate data...")
    
    # Main comparison table
    comparison_csv = os.path.join(TABLES_DIR, 'phase1_comparison_azure.csv')
    comparison_df.to_csv(comparison_csv)
    print(f"  Saved: {comparison_csv}")
    
    # Detailed results for each model
    detailed_csv = os.path.join(INTERMEDIATE_DIR, 'detailed_results.csv')
    if isinstance(detailed_results, dict):
        # Flatten detailed results
        flat_results = []
        for model_name, model_data in detailed_results.items():
            model_summary = model_data.copy() if isinstance(model_data, dict) else {}
            model_summary['Model'] = model_name
            flat_results.append(model_summary)
        detail_df = pd.DataFrame(flat_results)
    else:
        detail_df = detailed_results
    detail_df.to_csv(detailed_csv, index=False)
    print(f"  Saved: {detailed_csv}")
    
    # Test data with predictions for timeseries graph
    print("  Generating per-model predictions for timeseries...")
    test_data_copy = test_data.copy()
    
    feature_cols = [f'lag_{k}' for k in range(1, 11)]
    
    for model_name, model in baselines.items():
        pred_col = f'{model_name}_pred'
        test_data_copy[pred_col] = test_data_copy[feature_cols].apply(
            lambda row: model.predict(row.values), axis=1
        )
    
    timeseries_csv = os.path.join(INTERMEDIATE_DIR, 'timeseries_predictions.csv')
    test_data_copy.to_csv(timeseries_csv, index=False)
    print(f"  Saved: {timeseries_csv}")
    
    return comparison_df, test_data_copy


def generate_graphs(comparison_df, test_data_with_preds):
    """Generate publication-ready graphs."""
    print("\n[PHASE 1] Generating graphs...")
    
    # Extract key metrics (rename columns to match actual output)
    models = comparison_df.index.tolist()
    total_costs = comparison_df['Total Cost'].values
    overall_slas = comparison_df['SLA Compliance'].values  # Changed from 'Overall SLA'
    extreme_slas = comparison_df['SLA on Extreme'].values.astype(float)  # Changed from 'Extreme SLA'
    
    # Graph 1: Cost vs Models (Bar)
    print("  1. cost_vs_models.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, total_costs / 1e6, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Total Cost (Million)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Baseline Cost Comparison (Azure Test Set)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}M',
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'comparison', 'cost_vs_models.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph 2: SLA vs Models (Bar)
    print("  2. sla_vs_models.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, overall_slas, alpha=0.7, edgecolor='black')
    ax.set_ylabel('SLA Compliance (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Overall SLA Compliance (Azure Test Set)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'comparison', 'sla_vs_models.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph 3: Extreme SLA (Bar)
    print("  3. extreme_sla.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, extreme_slas, alpha=0.7, edgecolor='black', color='coral')
    ax.set_ylabel('Extreme Event SLA (p99+) (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('SLA on Extreme Events (p99+ Demand)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'comparison', 'extreme_sla.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph 4: Cost vs SLA Scatter
    print("  4. cost_vs_sla_scatter.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(extreme_slas, total_costs / 1e6, s=300, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (extreme_slas[i], total_costs[i] / 1e6),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Extreme Event SLA (p99+) (%)', fontsize=12)
    ax.set_ylabel('Total Cost (Million)', fontsize=12)
    ax.set_title('Cost-SLA Trade-off (Lower left is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'comparison', 'cost_vs_sla_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph 5: Timeseries (actual + baselines)
    print("  5. baseline_vs_actual.png")
    # Take first 500 timesteps for clarity
    window = min(500, len(test_data_with_preds))
    test_window = test_data_with_preds.iloc[:window].copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Actual demand
    ax.plot(range(window), test_window['concurrency'].values, 'k-', linewidth=2, label='Actual Demand', alpha=0.8)
    
    # Predictions from key baselines
    baseline_colors = {
        'Reactive': 'blue',
        'MLP Forecast': 'red',
        'Static (P90)': 'green',
    }
    
    for baseline_name, color in baseline_colors.items():
        pred_col = f'{baseline_name}_pred'
        if pred_col in test_window.columns:
            ax.plot(range(window), test_window[pred_col].values, color=color, 
                   linewidth=1.5, alpha=0.6, label=baseline_name, linestyle='--')
    
    ax.set_xlabel('Time (timesteps)', fontsize=12)
    ax.set_ylabel('Container Concurrency', fontsize=12)
    ax.set_title(f'Baseline Predictions vs Actual (First {window} timesteps)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'timeseries', 'baseline_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph 6: Demand Distribution (Histogram with p99)
    print("  6. demand_distribution.png")
    demand = test_data_with_preds['concurrency'].values
    p99_threshold = np.percentile(demand, 99)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    counts, bins, patches = ax.hist(demand, bins=50, alpha=0.7, edgecolor='black', label='Demand distribution')
    
    # Color bars for extreme region
    for i, patch in enumerate(patches):
        if bins[i] >= p99_threshold:
            patch.set_facecolor('coral')
            patch.set_alpha(0.8)
    
    # P99 line
    ax.axvline(p99_threshold, color='red', linestyle='--', linewidth=2, label=f'P99 threshold ({p99_threshold:.0f})')
    
    ax.set_xlabel('Container Concurrency', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Demand Distribution (Extreme Event Threshold)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'distribution', 'demand_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph 7: Extreme Events Over Time
    print("  7. extreme_events_plot.png")
    demand = test_data_with_preds['concurrency'].values
    p99_threshold = np.percentile(demand, 99)
    is_extreme = demand >= p99_threshold
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Time series with extreme events highlighted
    ax.plot(range(len(demand)), demand, 'b-', linewidth=1, alpha=0.6, label='Demand')
    
    # Mark extreme events
    extreme_indices = np.where(is_extreme)[0]
    ax.scatter(extreme_indices, demand[extreme_indices], color='red', s=50, 
              alpha=0.8, label='Extreme Events (p99+)', edgecolors='darkred', linewidth=1)
    
    # P99 line
    ax.axhline(p99_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P99 threshold')
    
    ax.set_xlabel('Time (timesteps)', fontsize=12)
    ax.set_ylabel('Container Concurrency', fontsize=12)
    ax.set_title(f'Extreme Events Over Time ({len(extreme_indices)} events, {100*len(extreme_indices)/len(demand):.1f}%)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'distribution', 'extreme_events_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  [OK] All 7 graphs generated successfully")


def main():
    """Main Phase 1 finalization workflow."""
    print("\n" + "="*90)
    print("PHASE 1 FINALIZATION: BASELINE EVALUATION & PUBLICATION GRAPHS")
    print("="*90)
    
    # Step 1: Load data
    train_data, test_data = load_and_prepare_data()
    if train_data is None:
        print("ERROR: Failed to load data")
        return
    
    # Step 2: Run comparison
    comparison_df, detailed_results, baselines, test_data = run_baseline_comparison(train_data, test_data)
    
    # Step 3: Save results
    comparison_df, test_data_with_preds = save_results_and_intermediate(
        comparison_df, detailed_results, test_data, baselines
    )
    
    # Step 4: Generate graphs
    generate_graphs(comparison_df, test_data_with_preds)
    
    # Summary
    print("\n" + "="*90)
    print("PHASE 1 FINALIZATION COMPLETE")
    print("="*90)
    print("\nResults Summary:")
    print(comparison_df.to_string())
    
    print("\n\nKey Deliverables:")
    print(f"  Tables:        results/phase1/tables/")
    print(f"  Graphs:        graphs/phase1/")
    print(f"  Intermediate:  results/phase1/intermediate/")
    
    print("\n" + "="*90)


if __name__ == '__main__':
    main()
