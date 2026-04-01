"""
Example evaluation script: Load test data, create dummy model, run evaluation.

This script demonstrates the complete evaluation pipeline:
    1. Load test dataset
    2. Create a simple baseline model
    3. Run evaluation
    4. Print results and comparison

Usage:
    python scripts/run_evaluation.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path so we can import evaluation module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.pipeline import run_evaluation
from evaluation.config import COST_COLD, COST_IDLE


# ============================================================================
# DUMMY MODEL FOR TESTING
# ============================================================================

class DummyModel:
    """
    Simple baseline: weighted average of lagged values.
    
    Weights: more recent lags get higher weight.
    This is a naive but reasonable baseline for autoscaling.
    """
    
    def __init__(self, lag_weight=0.8):
        """
        Initialize with exponential decay weights favoring recent lags.
        
        param lag_weight: decay factor for older lags (0 to 1)
        """
        self.lag_weight = lag_weight
        
        # Compute exponential weights for lags [1, 2, ..., 10]
        lags = np.arange(1, 11)
        self.weights = self.lag_weight ** (10 - lags)  # Recent lags = higher weight
        self.weights /= self.weights.sum()  # Normalize
    
    def predict(self, features):
        """
        Predict container count as weighted average of lags.
        
        INPUT: features, array of shape (10,) with [lag_1, lag_2, ..., lag_10]
        OUTPUT: predicted container count (float)
        """
        # Ensure features is properly formatted
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Weighted average
        prediction = np.dot(features, self.weights)
        
        return prediction


# ============================================================================
# SCRIPT: LOAD DATA AND RUN EVALUATION
# ============================================================================

def main():
    """
    Main script: Load test data and run evaluation on dummy model.
    """
    
    print("\n" + "="*70)
    print("COLD START OPTIMIZATION: EVALUATION FRAMEWORK TEST")
    print("="*70)
    
    # ====================================================================
    # Load test data (using Azure as example)
    # ====================================================================
    
    print("\n[SETUP] Loading test data...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_path = os.path.join(base_dir, 'dataset', 'processed', 'azure', 'test.csv')
    
    if not os.path.exists(test_data_path):
        print(f"  ERROR: Test data not found at {test_data_path}")
        print(f"  Please ensure preprocessing is complete.")
        return
    
    test_data = pd.read_csv(test_data_path)
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    
    print(f"  ✓ Loaded {len(test_data)} test samples from Azure dataset")
    print(f"    Columns: {list(test_data.columns)}")
    print(f"    Date range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
    
    # ====================================================================
    # Create dummy model
    # ====================================================================
    
    print("\n[SETUP] Creating baseline model...")
    
    model = DummyModel(lag_weight=0.85)
    print(f"  ✓ Initialized DummyModel (exponential-weighted lags)")
    print(f"    Lag weights: {model.weights[:3].round(3)} ... {model.weights[-1].round(3)}")
    
    # ====================================================================
    # Run evaluation
    # ====================================================================
    
    print("\n[EVAL] Starting evaluation pipeline...")
    
    output = run_evaluation(
        model=model,
        test_data=test_data,
        dataset_name='Azure Functions',
        model_name='DummyModel (Baseline)',
        cost_cold=COST_COLD,
        cost_idle=COST_IDLE,
        percentile_extreme=99,
    )
    
    # ====================================================================
    # Extract and display results
    # ====================================================================
    
    results_df = output['results_df']
    summary = output['summary']
    extreme_metrics = output['extreme_metrics']
    
    print("\n[RESULTS] Detailed Breakdown:")
    print(f"\n  Aggregate Metrics:")
    print(f"    Total Cold Starts:     {summary['total_cold']:>10,}")
    print(f"    Total Idle:            {summary['total_idle']:>10,}")
    print(f"    Total Cost:            {summary['total_cost']:>10,.2f}")
    print(f"    SLA Compliance:        {summary['sla_compliance']:>10.2%}")
    
    if extreme_metrics:
        print(f"\n  Extreme Event Analysis (p99):")
        print(f"    Extreme Timesteps:     {extreme_metrics['num_extreme_timesteps']:>10,} "
              f"({extreme_metrics['pct_extreme']:.1f}%)")
        print(f"    SLA on Extreme:        {extreme_metrics['sla_on_extreme']:>10.2%}")
        print(f"    Avg Cold Starts:       {extreme_metrics['cold_mean_on_extreme']:>10.1f}")
        print(f"    Max Cold Starts:       {extreme_metrics['cold_max_on_extreme']:>10.1f}")
        print(f"    Cost on Extreme:       {extreme_metrics['cost_total_on_extreme']:>10,.2f}")
    
    # ====================================================================
    # Show sample results
    # ====================================================================
    
    print(f"\n[SAMPLE] First 10 timesteps of results:")
    print(results_df[['timestamp', 'demand', 'containers', 'cold', 'idle', 'cost']].head(10).to_string())
    
    # ====================================================================
    # Summary statistics
    # ====================================================================
    
    print(f"\n[STATS] Demand statistics:")
    print(f"    Mean:      {results_df['demand'].mean():.1f}")
    print(f"    Median:    {results_df['demand'].median():.1f}")
    print(f"    Std Dev:   {results_df['demand'].std():.1f}")
    print(f"    P95:       {results_df['demand'].quantile(0.95):.1f}")
    print(f"    P99:       {results_df['demand'].quantile(0.99):.1f}")
    print(f"    Max:       {results_df['demand'].max():.1f}")
    
    # ====================================================================
    # Completion message
    # ====================================================================
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print("\nResults saved in memory. Next steps:")
    print("  1. Compare against other models using same pipeline")
    print("  2. Save results_df for plotting and analysis")
    print("  3. Run sensitivity analysis on cost parameters")
    print("  4. Aggregate results across all datasets")
    print()


if __name__ == '__main__':
    main()
