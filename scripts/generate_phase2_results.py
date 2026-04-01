import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import train_mlp_scaler
from evaluation.config import COST_COLD, COST_IDLE
from models.risk_aware import (
    train_quantile_models,
    fit_evt_model,
    RiskAwareScaler,
)

def evaluate_model_local(model, X, y, cost_cold=COST_COLD, cost_idle=COST_IDLE):
    """Evaluate a model without relying on the external pipeline logger to avoid I/O noise."""
    preds = []
    for i in range(len(X)):
        p = model.predict(X[i])
        preds.append(p)
    preds = np.array(preds)
    
    cold = np.maximum(0, y - preds)
    idle = np.maximum(0, preds - y)
    
    cost = np.sum(cost_cold * cold + cost_idle * idle)
    total_cold = np.sum(cold)
    total_idle = np.sum(idle)
    
    # SLA: percentage of timesteps with ZERO cold starts
    sla = np.mean(cold == 0)
    
    return {
        'predictions': preds,
        'cost': cost,
        'cold': total_cold,
        'idle': total_idle,
        'sla': sla
    }

def generate_results():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'azure', 'train.csv')
    test_path = os.path.join(base_dir, 'data', 'processed', 'azure', 'test.csv')
    
    print("[1/3] Loading Data & Training Models...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    feature_cols = [f'lag_{k}' for k in range(1, 11)]
    X_test = test_data[feature_cols].values
    y_test = test_data['concurrency'].values
    
    # Identify extreme events (Test set)
    thresh = np.percentile(y_test, 99)
    extreme_idx = np.where(y_test >= thresh)[0]
    print(f"  Extreme threshold (Q99 of test): {thresh:.2f} ({len(extreme_idx)} events)")
    
    # Train Models
    mlp_baseline = train_mlp_scaler(train_data, verbose=False)
    q_models = train_quantile_models(train_data, verbose=False)
    evt_params = fit_evt_model(train_data['concurrency'].values, verbose=False)
    
    risk_aware = RiskAwareScaler(
        quantile_models=q_models,
        evt_params=evt_params,
        alpha=0.01,
        n_scenarios=300,
        cost_cold=COST_COLD,
        cost_idle=COST_IDLE,
    )
    
    print("[2/3] Evaluating Models...")
    models = {'MLPForecast': mlp_baseline, 'RiskAware (V2)': risk_aware}
    results = {}
    
    avg_demand = np.mean(y_test)
    
    for name, model in models.items():
        print(f"  Running {name}...")
        res = evaluate_model_local(model, X_test, y_test)
        
        # Extreme SLA
        ext_cold = np.maximum(0, y_test[extreme_idx] - res['predictions'][extreme_idx])
        ext_sla = np.mean(ext_cold == 0) if len(extreme_idx) > 0 else 1.0
        
        # Provisioning ratio
        avg_containers = np.mean(res['predictions'])
        prov_ratio = avg_containers / avg_demand
        
        results[name] = {
            'Model': name,
            'Total Cost ($)': res['cost'],
            'Total Cold Starts': res['cold'],
            'Total Idle Capacity': res['idle'],
            'Overall SLA (%)': res['sla'] * 100,
            'Extreme SLA (%)': ext_sla * 100,
            'Provisioning Ratio (x)': prov_ratio
        }
        
        # Save predictions for plotting
        if name == 'MLPForecast':
            test_data['mlp_pred'] = res['predictions']
        elif name == 'RiskAware (V2)':
            test_data['risk_pred'] = res['predictions']
    
    df_res = pd.DataFrame(list(results.values()))
    
    # Save CSV
    out_csv = os.path.join(base_dir, 'results', 'phase2', 'tables', 'phase2_comparison_azure.csv')
    df_res.to_csv(out_csv, index=False)
    print(f"  Saved tabular results to {out_csv}")
    
    print("[3/3] Generating Graphs...")
    
    # Setup graph paths
    g_dir = os.path.join(base_dir, 'graphs', 'phase2')
    
    # 1. Cost vs Model
    plt.figure(figsize=(8, 6))
    plt.bar(df_res['Model'], df_res['Total Cost ($)'] / 1e6, width=0.5, zorder=3)
    plt.ylabel('Total Cost (Millions $)')
    plt.title('Total Operational Cost (Azure)')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.savefig(os.path.join(g_dir, 'comparison', 'cost_vs_model.png'), bbox_inches='tight')
    plt.close()
    
    # 2. SLA vs Model
    plt.figure(figsize=(8, 6))
    plt.bar(df_res['Model'], df_res['Overall SLA (%)'], width=0.5, zorder=3)
    plt.ylabel('Overall SLA Compliance (%)')
    plt.title('Overall SLA (Azure)')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.savefig(os.path.join(g_dir, 'comparison', 'sla_vs_model.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Extreme SLA
    plt.figure(figsize=(8, 6))
    plt.bar(df_res['Model'], df_res['Extreme SLA (%)'], width=0.5, zorder=3)
    plt.ylabel('Extreme SLA Compliance (%)')
    plt.title('SLA During Top 1% Demand Spikes (Azure)')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.savefig(os.path.join(g_dir, 'comparison', 'extreme_sla.png'), bbox_inches='tight')
    plt.close()
    
    # 4. Scatter Cost vs SLA
    plt.figure(figsize=(8, 6))
    plt.scatter(df_res['Total Cost ($)'] / 1e6, df_res['Overall SLA (%)'], s=150, zorder=3)
    for i, row in df_res.iterrows():
        plt.annotate(row['Model'], (row['Total Cost ($)'] / 1e6, row['Overall SLA (%)']), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Total Cost (Millions $)')
    plt.ylabel('Overall SLA Compliance (%)')
    plt.title('Cost vs. Reliability Pareto Frontier')
    plt.grid(linestyle='--', alpha=0.7, zorder=0)
    plt.savefig(os.path.join(g_dir, 'comparison', 'cost_vs_sla_scatter.png'), bbox_inches='tight')
    plt.close()
    
    # 5. Timeseries (Prediction vs Actual) -> find a 100-step window with a spike
    window_start = max(0, extreme_idx[0] - 50 if len(extreme_idx) > 0 else 1000)
    window_end = window_start + 100
    
    plt.figure(figsize=(12, 5))
    plt.plot(test_data['concurrency'].iloc[window_start:window_end].values, label='Actual Demand', color='black', linewidth=1.5)
    plt.plot(test_data['mlp_pred'].iloc[window_start:window_end].values, label='MLP Forecast', linestyle='--')
    plt.plot(test_data['risk_pred'].iloc[window_start:window_end].values, label='Risk-Aware (V2)', linestyle='-.', linewidth=2)
    plt.ylabel('Container Demand')
    plt.xlabel('Timestep (Relative)')
    plt.title('Demand Trajectory vs Provisioning Decisions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(g_dir, 'timeseries', 'prediction_vs_actual.png'), bbox_inches='tight')
    plt.close()
    
    # 6. Ablation V1 vs V2 Cost
    # Values from previous validated run
    v1_cost = 717585982.0 
    v2_cost = results['RiskAware (V2)']['Total Cost ($)']
    
    plt.figure(figsize=(6, 5))
    plt.bar(['V1 (Always-On EVT)', 'V2 (Conditional EVT)'], [v1_cost/1e6, v2_cost/1e6], zorder=3)
    plt.ylabel('Total Cost (Millions $)')
    plt.title('Ablation: Cost Impact of Conditional Gating')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.savefig(os.path.join(g_dir, 'ablation', 'v1_vs_v2_cost.png'), bbox_inches='tight')
    plt.close()
    
    # 7. Ablation V1 vs V2 SLA
    v1_sla = 99.95
    v2_sla = results['RiskAware (V2)']['Overall SLA (%)']
    
    plt.figure(figsize=(6, 5))
    plt.bar(['V1 (Always-On EVT)', 'V2 (Conditional EVT)'], [v1_sla, v2_sla], zorder=3)
    plt.ylabel('Overall SLA (%)')
    plt.title('Ablation: SLA Retention after Conditional Gating')
    plt.ylim(90, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.savefig(os.path.join(g_dir, 'ablation', 'v1_vs_v2_sla.png'), bbox_inches='tight')
    plt.close()
    
    print("  All graphs generated successfully.")

if __name__ == "__main__":
    generate_results()
