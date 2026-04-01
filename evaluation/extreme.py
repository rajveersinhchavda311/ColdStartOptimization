"""
Extreme event identification and analysis.

Functions:
    identify_extreme_events() - Flag extreme demand timesteps (p99+)
    analyze_extreme_events() - Compute metrics on extreme subset
"""

import pandas as pd


def identify_extreme_events(results_df, percentile=99):
    """
    Flag extreme demand timesteps in results DataFrame.
    
    Extreme = demand >= p{percentile}
    This identifies the most challenging scenarios (top 1% by default).
    
    INPUT:
    ------
    results_df : DataFrame from evaluate_model()
    percentile : int, threshold for extreme (default 99 = top 1%)
    
    OUTPUT:
    -------
    results_df : DataFrame with new column 'is_extreme' (bool)
    demand_threshold : float, the actual p{percentile} value
    """
    
    # Compute percentile threshold
    demand_threshold = results_df['demand'].quantile(percentile / 100.0)
    
    # Flag rows at or above threshold
    results_df['is_extreme'] = results_df['demand'] >= demand_threshold
    
    return results_df, demand_threshold


def analyze_extreme_events(results_df):
    """
    Compute metrics specifically on extreme timesteps.
    
    Answers: How does the model perform when demand is highest?
    
    INPUT:
    ------
    results_df : DataFrame (must have 'is_extreme' column from identify_extreme_events)
    
    OUTPUT:
    -------
    dict with metrics on extreme subset, or None if no extreme events
    """
    
    extreme_df = results_df[results_df['is_extreme']]
    
    if len(extreme_df) == 0:
        return None
    
    metrics = {
        'num_extreme_timesteps': int(len(extreme_df)),
        'pct_extreme': float(100.0 * len(extreme_df) / len(results_df)),
        'demand_threshold': float(extreme_df['demand'].min()),
        
        'cold_mean_on_extreme': float(extreme_df['cold'].mean()),
        'cold_max_on_extreme': float(extreme_df['cold'].max()),
        'cold_p95_on_extreme': float(extreme_df['cold'].quantile(0.95)),
        
        'idle_mean_on_extreme': float(extreme_df['idle'].mean()),
        'idle_p95_on_extreme': float(extreme_df['idle'].quantile(0.95)),
        
        'cost_total_on_extreme': float(extreme_df['cost'].sum()),
        'cost_mean_on_extreme': float(extreme_df['cost'].mean()),
        
        'sla_on_extreme': float((extreme_df['cold'] == 0).sum() / len(extreme_df)),
    }
    
    return metrics
