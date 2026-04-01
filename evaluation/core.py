"""
Core evaluation logic: Model simulation and metric computation.

Main function:
    evaluate_model() - Run model on test set and compute metrics
"""

import pandas as pd
import numpy as np


def evaluate_model(model, test_data, cost_cold=10.0, cost_idle=1.0):
    """
    Simulate model on test set and compute cost metrics.
    
    LOGIC (NO DATA LEAKAGE):
        For each timestep t:
            1. Extract features (lag_1 ... lag_10) from historical data
            2. Model predicts container count
            3. Observe actual demand (concurrency)
            4. Compute: cold starts, idle, cost
            5. Append to results
    
    INPUTS:
    -------
    model : object with predict(features) method
        predict() takes 1D array of shape (10,) or (n, 10).
        Returns numeric container count.
    
    test_data : pandas.DataFrame
        Columns MUST include: ['timestamp', 'concurrency', 'lag_1', ..., 'lag_10']
        Rows in chronological order, no NaN values in lag columns.
    
    cost_cold : float
        Penalty per cold start (unserved request).
    
    cost_idle : float
        Cost per idle container-second.
    
    OUTPUTS:
    --------
    results_df : pandas.DataFrame
        Columns: ['timestamp', 'demand', 'containers', 'cold', 'idle', 'cost']
        One row per test timestep.
    
    summary_metrics : dict
        Keys: ['total_cold', 'total_idle', 'total_cost', 'sla_compliance']
    """
    
    # Validation
    required_cols = {'timestamp', 'concurrency'} | {f'lag_{k}' for k in range(1, 11)}
    assert required_cols.issubset(test_data.columns), (
        f"test_data missing columns. Required: {required_cols}"
    )
    assert 'NaN' not in test_data[['lag_{}'.format(k) for k in range(1, 11)]].values, (
        "NaN values found in lag features"
    )
    
    # Initialize result collector
    results_list = []
    
    # ========================================================================
    # EVALUATION LOOP (simulation)
    # ========================================================================
    
    for idx, row in test_data.iterrows():
        
        # Step 1: Extract features (historical lags only)
        feature_cols = [f'lag_{k}' for k in range(1, 11)]
        features = row[feature_cols].values  # shape (10,)
        
        # Step 2: Model predicts container count
        predicted_raw = model.predict(features)
        predicted_containers = max(0, int(round(predicted_raw)))
        
        # Step 3: Ground truth demand
        actual_demand = int(row['concurrency'])
        
        # Step 4: Compute per-timestep metrics
        cold = max(0, actual_demand - predicted_containers)
        idle = max(0, predicted_containers - actual_demand)
        cost = cost_idle * idle + cost_cold * cold
        
        # Step 5: Append result
        results_list.append({
            'timestamp': row['timestamp'],
            'demand': actual_demand,
            'containers': predicted_containers,
            'cold': cold,
            'idle': idle,
            'cost': cost,
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Compute summary metrics
    summary_metrics = {
        'total_cold': int(results_df['cold'].sum()),
        'total_idle': int(results_df['idle'].sum()),
        'total_cost': float(results_df['cost'].sum()),
        'sla_compliance': float((results_df['cold'] == 0).sum() / len(results_df)),
    }
    
    return results_df, summary_metrics


def aggregate_results(results_df, cost_cold=10.0, cost_idle=1.0):
    """
    Compute summary metrics from results DataFrame.
    
    INPUT: results_df from evaluate_model()
    OUTPUT: dict with summary metrics
    """
    
    metrics = {
        'total_cold': int(results_df['cold'].sum()),
        'total_idle': int(results_df['idle'].sum()),
        'total_cost': float(results_df['cost'].sum()),
        'sla_compliance': float((results_df['cold'] == 0).sum() / len(results_df)),
    }
    
    return metrics


def distribution_stats(results_df):
    """
    Compute percentile statistics for understanding metric distributions.
    
    INPUT: results_df from evaluate_model()
    OUTPUT: dict with percentiles and extrema
    """
    
    stats = {
        'cold_p50': float(results_df['cold'].quantile(0.50)),
        'cold_p95': float(results_df['cold'].quantile(0.95)),
        'cold_p99': float(results_df['cold'].quantile(0.99)),
        'cold_max': float(results_df['cold'].max()),
        
        'idle_p50': float(results_df['idle'].quantile(0.50)),
        'idle_p95': float(results_df['idle'].quantile(0.95)),
        'idle_p99': float(results_df['idle'].quantile(0.99)),
        'idle_max': float(results_df['idle'].max()),
        
        'cost_mean': float(results_df['cost'].mean()),
        'cost_std': float(results_df['cost'].std()),
    }
    
    return stats
