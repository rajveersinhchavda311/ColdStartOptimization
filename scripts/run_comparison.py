"""
Example script: Baseline model comparison.

This script demonstrates the model comparison pipeline:
    1. Load train and test datasets
    2. Create baseline models from training data
    3. Run fair comparison on test set
    4. Print results table

Usage:
    python scripts/run_comparison.py
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import create_baselines, compare_models, print_comparison, train_tcn_scaler
from evaluation.config import COST_COLD, COST_IDLE


def main():
    """
    Main script: Load data, create baselines, run comparison.
    """
    
    print("\n" + "="*90)
    print("BASELINE MODEL COMPARISON")
    print("="*90)
    
    # ====================================================================
    # Load data
    # ====================================================================
    
    print("\n[SETUP] Loading datasets...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load train data (used to initialize baselines)
    train_path = os.path.join(base_dir, 'dataset', 'processed', 'azure', 'train.csv')
    test_path = os.path.join(base_dir, 'dataset', 'processed', 'azure', 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"  ERROR: Data files not found")
        print(f"    Train: {train_path}")
        print(f"    Test:  {test_path}")
        return
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Parse timestamps
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    
    print(f"  [OK] Loaded train data:  {len(train_data)} samples")
    print(f"  [OK] Loaded test data:   {len(test_data)} samples")
    
    # ====================================================================
    # Create baseline models from training data
    # ====================================================================
    
    print("\n[SETUP] Creating baseline models from training data...")
    
    baselines = create_baselines(train_data)
    
    print(f"  [OK] Created {len(baselines)} baseline models:")
    for name in baselines.keys():
        print(f"    - {name}")
    
    # Train TCN forecasting baseline
    print("\n[SETUP] Training TCN forecasting baseline...")
    tcn_scaler = train_tcn_scaler(train_data, verbose=True)
    baselines['TCN Forecast'] = tcn_scaler
    print(f"  [OK] Added TCN Forecast to baselines (total: {len(baselines)})")
    
    # ====================================================================
    # Run comparison on test set
    # ====================================================================
    
    print("\n[EVAL] Running fair comparison on test set...")
    print(f"      Cost parameters: C_cold={COST_COLD}, C_idle={COST_IDLE}")
    
    comparison_df, detailed_results = compare_models(
        models_dict=baselines,
        test_data=test_data,
        dataset_name='Azure Functions (Test Set)',
        cost_cold=COST_COLD,
        cost_idle=COST_IDLE,
        percentile_extreme=99,
        verbose=False,  # Suppress per-model verbose output
    )
    
    # ====================================================================
    # Print results
    # ====================================================================
    
    print("\n[RESULTS] Baseline Model Comparison\n")
    
    # Format for display
    print("\nFULL RESULTS:")
    print(comparison_df)
    
    # Compute and display ranking
    print("\n" + "="*90)
    print("RANKING BY TOTAL COST (Lower is Better)")
    print("="*90)
    
    # Extract numeric values
    cost_values = []
    for idx in comparison_df.index:
        cost = comparison_df.loc[idx, 'Total Cost']
        try:
            # Handle formatted strings
            if isinstance(cost, str):
                cost = float(cost.replace(',', ''))
        except:
            pass
        cost_values.append((idx, float(cost)))
    
    cost_values.sort(key=lambda x: x[1])
    
    for rank, (model_name, cost) in enumerate(cost_values, 1):
        sla = comparison_df.loc[model_name, 'SLA Compliance']
        try:
            if isinstance(sla, str):
                sla = float(sla.strip('%')) / 100
        except:
            pass
        
        print(f"  {rank}. {model_name:25s} | Cost: {cost:>13,.2f} | SLA: {sla:>6.2%}")
    
    # ====================================================================
    # Summary statistics
    # ====================================================================
    
    print("\n" + "="*90)
    print("SUMMARY STATISTICS")
    print("="*90)
    
    # Calculate cost delta from best
    cost_values_dict = {name: cost for name, cost in cost_values}
    best_cost = cost_values[0][1]
    
    print(f"\nBest Model (by cost): {cost_values[0][0]}")
    print(f"Best Cost: {best_cost:,.2f}\n")
    
    for rank, (model_name, cost) in enumerate(cost_values, 1):
        delta_pct = 100 * (cost - best_cost) / best_cost
        delta_abs = cost - best_cost
        
        extreme_sla = detailed_results[model_name]['extreme_metrics'].get('sla_on_extreme', float('nan'))
        
        print(f"{rank}. {model_name:25s} | Delta: {delta_pct:>+6.1f}% ({delta_abs:>+10,.0f}) | Extreme SLA: {extreme_sla:>6.2%}")
    
    # ====================================================================
    # Key insights
    # ====================================================================
    
    print("\n" + "="*90)
    print("KEY INSIGHTS")
    print("="*90)
    
    print("\nModel Behavior:")
    for model_name, details in detailed_results.items():
        summary = details['summary']
        extreme = details['extreme_metrics']
        
        print(f"\n{model_name}:")
        print(f"  Overall SLA:    {summary['sla_compliance']:.2%}")
        print(f"  Extreme SLA:    {extreme.get('sla_on_extreme', 0):.2%} "
              f"(degradation: {summary['sla_compliance'] - extreme.get('sla_on_extreme', 0):.2%})")
        print(f"  Cold/Idle Ratio: {summary['total_cold'] / max(summary['total_idle'], 1):.2f}")
    
    print("\n" + "="*90)
    print("✓ COMPARISON COMPLETE")
    print("="*90 + "\n")
    
    return comparison_df, detailed_results


if __name__ == '__main__':
    comparison_df, detailed_results = main()
