import os
import sys
import traceback
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import create_baselines, train_mlp_scaler
from evaluation.config import COST_COLD, COST_IDLE
from models.risk_aware import (
    train_quantile_models,
    fit_evt_model,
    generate_scenarios,
    optimize_cvar,
    RiskAwareScaler,
)

def analyze_validation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'dataset', 'processed', 'azure', 'train.csv')
    test_path = os.path.join(base_dir, 'dataset', 'processed', 'azure', 'test.csv')
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    
    feature_cols = [f'lag_{k}' for k in range(1, 11)]
    X_test = test_data[feature_cols].values
    y_test = test_data['concurrency'].values
    
    print("[1/2] Training models...")
    # MLP Baseline
    mlp_baseline = train_mlp_scaler(train_data, verbose=False)
    
    # Risk-Aware Model Components
    q_models = train_quantile_models(train_data, verbose=False)
    evt_params = fit_evt_model(train_data['concurrency'].values, verbose=False)
    
    risk_aware = RiskAwareScaler(
        quantile_models=q_models,
        evt_params=evt_params,
        alpha=0.05,
        n_scenarios=300,
        cost_cold=COST_COLD,
        cost_idle=COST_IDLE,
    )
    
    print("[2/2] Running analysis...")
    
    # --- TASK 1 & 2: Decision Sanity Check & Baseline Comparison ---
    sample_size = 30
    start_idx = 0 # Beginning of test set
    
    results_30 = []
    for i in range(start_idx, start_idx + sample_size):
        features = X_test[i]
        actual = y_test[i]
        
        mlp_pred = mlp_baseline.predict(features)
        risk_pred = risk_aware.predict(features)
        
        cold = max(0, actual - risk_pred)
        idle = max(0, risk_pred - actual)
        
        results_30.append({
            'timestep': i,
            'actual': actual,
            'risk_pred': risk_pred,
            'mlp_pred': mlp_pred,
            'cold': cold,
            'idle': idle
        })
        
    df_30 = pd.DataFrame(results_30)
    
    print("\n" + "="*80)
    print("TASK 1 & 2: DECISION SANITY CHECK & BASELINE COMPARISON (First 30 steps)")
    print("="*80)
    print(df_30[['timestep', 'actual', 'risk_pred', 'mlp_pred', 'cold', 'idle']].to_string(index=False))
    
    # --- TASK 3: EVT ACTIVATION CHECK ---
    # In the current implementation, generate_scenarios ALWAYS includes EVT tail.
    # However, we can measure how often the predicted c* > predicted q99.
    # If c* > q99, it means the EVT tail (which starts at u=P99_train) or the q99 forecast itself
    # was significant enough to pull the CVaR above the body.
    
    evt_influence_count = 0
    total_steps = len(test_data)
    
    all_risk_preds = []
    
    # Efficient loop for task 3 & 5
    for i in range(total_steps):
        features = X_test[i]
        # Predict quantiles to see "baseline"
        f2d = features.reshape(1, -1)
        q99 = float(q_models[0.99].predict(f2d)[0])
        
        risk_pred = risk_aware.predict(features)
        all_risk_preds.append(risk_pred)
        
        if risk_pred > q99:
            evt_influence_count += 1
            
    print("\n" + "="*80)
    print("TASK 3: EVT ACTIVATION / INFLUENCE CHECK")
    print("="*80)
    print(f"Total test steps: {total_steps}")
    print(f"Steps where RiskPred > Q99: {evt_influence_count}")
    print(f"Percentage of EVT-Influenced steps: {100 * evt_influence_count / total_steps:.2f}%")
    
    # --- TASK 4: SCENARIO DISTRIBUTION CHECK ---
    print("\n" + "="*80)
    print("TASK 4: SCENARIO DISTRIBUTION CHECK")
    print("="*80)
    
    selected_indices = [0, 1000] # Use index 0 and another one later
    for idx in selected_indices:
        features = X_test[idx]
        f2d = features.reshape(1, -1)
        q50 = float(q_models[0.50].predict(f2d)[0])
        q90 = float(q_models[0.90].predict(f2d)[0])
        q99 = float(q_models[0.99].predict(f2d)[0])
        q90 = max(q90, q50)
        q99 = max(q99, q90)
        
        scenarios = generate_scenarios(q50, q90, q99, evt_params, n_scenarios=300, rng=np.random.default_rng(42))
        
        print(f"\nTimestep {idx}:")
        print(f"  Quantile Forecasts: Q50={q50:,.0f}, Q90={q90:,.0f}, Q99={q99:,.0f}")
        print(f"  Scenario Stats:")
        print(f"    Min:  {np.min(scenarios):,.0f}")
        print(f"    Mean: {np.mean(scenarios):,.0f}")
        print(f"    Max:  {np.max(scenarios):,.0f}")
        print(f"    P50:  {np.percentile(scenarios, 50):,.0f}")
        print(f"    P90:  {np.percentile(scenarios, 90):,.0f}")
        print(f"    P99:  {np.percentile(scenarios, 99):,.0f}")

    # --- TASK 5: COST BEHAVIOR ANALYSIS ---
    all_risk_preds = np.array(all_risk_preds)
    avg_demand = np.mean(y_test)
    avg_containers = np.mean(all_risk_preds)
    
    total_cold = np.sum(np.maximum(0, y_test - all_risk_preds))
    total_idle = np.sum(np.maximum(0, all_risk_preds - y_test))
    
    cold_cost = total_cold * COST_COLD
    idle_cost = total_idle * COST_IDLE
    total_cost = cold_cost + idle_cost
    
    print("\n" + "="*80)
    print("TASK 5: COST BEHAVIOR ANALYSIS")
    print("="*80)
    print(f"Average Demand:         {avg_demand:,.0f}")
    print(f"Average Containers:     {avg_containers:,.0f}")
    print(f"Provisioning Ratio:     {avg_containers/avg_demand:.2f}x")
    print(f"Total Cold Starts:      {total_cold:,.0f}")
    print(f"Total Idle Capacity:    {total_idle:,.0f}")
    print(f"Cold/Idle Ratio:        {total_cold/total_idle:.4f}")
    print(f"Cost Breakdown:")
    print(f"  Cold Cost (x{COST_COLD}): {cold_cost:,.0f} ({100*cold_cost/total_cost:.1f}%)")
    print(f"  Idle Cost (x{COST_IDLE}): {idle_cost:,.0f} ({100*idle_cost/total_cost:.1f}%)")
    print(f"Total Cost:             {total_cost:,.0f}")

if __name__ == "__main__":
    try:
        analyze_validation()
    except Exception as e:
        print("\n" + "!"*80)
        print("ERROR IN VALIDATION SCRIPT:")
        print("!"*80)
        traceback.print_exc()
        sys.exit(1)
