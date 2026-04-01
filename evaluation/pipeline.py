"""
End-to-end evaluation orchestration.

Main function:
    run_evaluation() - Ties together core + extreme analysis
"""

from .core import evaluate_model, distribution_stats
from .extreme import identify_extreme_events, analyze_extreme_events


def run_evaluation(model, test_data, dataset_name='Dataset', model_name='Model',
                   cost_cold=10.0, cost_idle=1.0, percentile_extreme=99):
    """
    End-to-end evaluation: simulate model, compute metrics, identify extremes.
    
    INPUT:
    ------
    model : object with predict(features) method
    test_data : DataFrame with ['timestamp', 'concurrency', 'lag_1', ..., 'lag_10']
    dataset_name : str, name of dataset (for logging)
    model_name : str, name of model (for logging)
    cost_cold : float, cold start penalty
    cost_idle : float, idle container cost
    percentile_extreme : int, percentile for extreme event definition
    
    OUTPUT:
    -------
    output : dict with keys:
        - results_df: Full results DataFrame with 'is_extreme' column
        - summary: Aggregate metrics (total_cold, total_idle, total_cost, sla_compliance)
        - distribution: Percentile statistics
        - extreme_metrics: Metrics on extreme subset
        - demand_threshold: P{percentile_extreme} demand value
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name:20s} on {dataset_name}")
    print(f"{'='*70}")
    
    # ========================================================================
    # Step 1: Simulate model on test set
    # ========================================================================
    
    print(f"\n[1/4] Running simulation ({len(test_data)} timesteps)...")
    results_df, summary = evaluate_model(model, test_data, cost_cold, cost_idle)
    print(f"      ✓ Computed metrics for {len(results_df)} timesteps")
    
    # ========================================================================
    # Step 2: Print summary metrics
    # ========================================================================
    
    print(f"\n[2/4] Summary Metrics:")
    print(f"      Total Cold Starts: {summary['total_cold']:,}")
    print(f"      Total Idle:        {summary['total_idle']:,}")
    print(f"      Total Cost:        {summary['total_cost']:,.2f}")
    print(f"      SLA Compliance:    {summary['sla_compliance']:.2%}")
    
    # Compute and display distribution statistics
    dist_stats = distribution_stats(results_df)
    print(f"\n      Cold Start Distribution:")
    print(f"        P50: {dist_stats['cold_p50']:.1f}, "
          f"P95: {dist_stats['cold_p95']:.1f}, "
          f"P99: {dist_stats['cold_p99']:.1f}, "
          f"Max: {dist_stats['cold_max']:.1f}")
    
    # ========================================================================
    # Step 3: Identify and analyze extreme events
    # ========================================================================
    
    print(f"\n[3/4] Extreme Event Analysis (p{percentile_extreme})...")
    results_df, threshold = identify_extreme_events(results_df, percentile=percentile_extreme)
    extreme_metrics = analyze_extreme_events(results_df)
    
    if extreme_metrics:
        print(f"      Demand Threshold: {threshold:.1f}")
        print(f"      Extreme Timesteps: {extreme_metrics['num_extreme_timesteps']:,} "
              f"({extreme_metrics['pct_extreme']:.1f}%)")
        print(f"      SLA on Extreme: {extreme_metrics['sla_on_extreme']:.2%}")
        print(f"      Avg Cold Starts on Extreme: {extreme_metrics['cold_mean_on_extreme']:.1f}")
    else:
        print(f"      No extreme events found")
        extreme_metrics = {}
    
    # ========================================================================
    # Step 4: Package results
    # ========================================================================
    
    print(f"\n[4/4] Finalizing results...")
    
    output = {
        'results_df': results_df,
        'summary': summary,
        'distribution': dist_stats,
        'extreme_metrics': extreme_metrics,
        'demand_threshold': threshold,
    }
    
    print(f"      ✓ Ready for analysis and comparison")
    print(f"{'='*70}\n")
    
    return output
