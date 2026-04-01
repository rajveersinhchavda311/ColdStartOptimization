"""
Multi-model evaluation and comparison.

Functions:
    compare_models() - Run evaluation on multiple models and produce summary table
    format_comparison_table() - Pretty-print comparison results
"""

import pandas as pd
from .pipeline import run_evaluation


def compare_models(models_dict, test_data, dataset_name='Dataset',
                   cost_cold=10.0, cost_idle=1.0, percentile_extreme=99,
                   verbose=True):
    """
    Evaluate multiple models on identical test set and produce comparison table.
    
    CRITICAL: All models are evaluated on the SAME test data with SAME cost params.
    This ensures fair direct comparison.
    
    INPUT:
    ------
    models_dict : dict
        Keys: model names (str)
        Values: model objects with predict(features) method
    
    test_data : pd.DataFrame
        Test set with ['timestamp', 'concurrency', 'lag_1', ..., 'lag_10']
        SAME for all models (fair comparison).
    
    dataset_name : str
        Name of dataset (for logging).
    
    cost_cold : float
        Cost per cold start (same for all models).
    
    cost_idle : float
        Cost per idle container (same for all models).
    
    percentile_extreme : int
        Percentile threshold for extreme events.
    
    verbose : bool
        If True, print progress for each model.
    
    OUTPUT:
    -------
    comparison_df : pd.DataFrame
        Summary metrics for each model (one row per model).
        Columns: model name + all summary metrics + extreme metrics
    
    detailed_results : dict
        Keys: model names
        Values: full output dict from run_evaluation() for each model
    """
    
    detailed_results = {}
    comparison_rows = []
    
    # ========================================================================
    # EVALUATE EACH MODEL
    # ========================================================================
    
    for model_name, model in models_dict.items():
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*70}")
        
        # Run evaluation
        output = run_evaluation(
            model=model,
            test_data=test_data,
            dataset_name=dataset_name,
            model_name=model_name,
            cost_cold=cost_cold,
            cost_idle=cost_idle,
            percentile_extreme=percentile_extreme,
        )
        
        # Store detailed results for later analysis
        detailed_results[model_name] = output
        
        # Extract summary metrics
        summary = output['summary']
        extreme = output.get('extreme_metrics', {})
        
        # Build row for comparison table
        row = {
            'Model': model_name,
            'Total Cold': summary['total_cold'],
            'Total Idle': summary['total_idle'],
            'Total Cost': summary['total_cost'],
            'SLA Compliance': summary['sla_compliance'],
        }
        
        # Add extreme event metrics if available
        if extreme:
            row['SLA on Extreme'] = extreme.get('sla_on_extreme', float('nan'))
            row['Cold Mean (Extreme)'] = extreme.get('cold_mean_on_extreme', float('nan'))
            row['Avg Cost (Extreme)'] = extreme.get('cost_total_on_extreme', 0) / extreme.get('num_extreme_timesteps', 1)
        
        comparison_rows.append(row)
    
    # ========================================================================
    # BUILD COMPARISON TABLE
    # ========================================================================
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Set model as index for cleaner display
    comparison_df = comparison_df.set_index('Model')
    
    return comparison_df, detailed_results


def format_comparison_table(comparison_df, decimal_places=2):
    """
    Format comparison DataFrame for printing.
    
    INPUT:
    ------
    comparison_df : pd.DataFrame
        Output from compare_models()
    
    decimal_places : int
        Number of decimal places for floating-point values
    
    OUTPUT:
    -------
    formatted : pd.DataFrame
        DataFrame with nicely formatted values
    """
    
    formatted = comparison_df.copy()
    
    # Format columns
    for col in formatted.columns:
        if 'Cost' in col or 'Idle' in col or 'Cold' in col:
            # Large numbers: format with commas or thousands
            if formatted[col].dtype in ['int64', 'float64']:
                formatted[col] = formatted[col].apply(
                    lambda x: f"{x:,.0f}" if x > 100 else f"{x:.{decimal_places}f}"
                )
        
        elif 'Compliance' in col or 'Extreme' in col:
            # Percentages/fractions
            if formatted[col].dtype in ['float64']:
                formatted[col] = formatted[col].apply(
                    lambda x: f"{x:.1%}" if x < 2 else f"{x:.{decimal_places}f}"
                )
    
    return formatted


def print_comparison(comparison_df, title='Model Comparison', verbose=False):
    """
    Pretty-print comparison table.
    
    INPUT:
    ------
    comparison_df : pd.DataFrame
        Output from compare_models()
    
    title : str
        Title for the table
    
    verbose : bool
        If True, print additional statistics
    """
    
    print(f"\n{'='*90}")
    print(f"{title:^90}")
    print(f"{'='*90}\n")
    
    print(comparison_df.to_string())
    
    print(f"\n{'='*90}\n")
    
    # Optional: print ranking by cost
    if verbose and 'Total Cost' in comparison_df.columns:
        print("RANKING BY TOTAL COST:")
        cost_col = 'Total Cost'
        
        # Try to convert back to numeric if formatted
        try:
            numeric_costs = pd.to_numeric(
                comparison_df[cost_col].astype(str).str.replace(',', ''),
                errors='coerce'
            )
        except:
            numeric_costs = comparison_df[cost_col]
        
        ranking = numeric_costs.sort_values().index
        for rank, model_name in enumerate(ranking, 1):
            print(f"  {rank}. {model_name}")
