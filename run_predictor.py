import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import genpareto
import joblib
from tcn import TCN
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting the final ROBUST prediction, simulation, and plotting pipeline...")

# --- 1. CONFIGURATION AND LOADING MODELS ---
print("Loading all model components...")
custom_objects = {'TCN': TCN}
tcn_model = tf.keras.models.load_model('best_tcn_model_all_regions.keras', custom_objects=custom_objects, compile=False)
evt_model = joblib.load('evt_model_all_regions.joblib')

# --- Parameters ---
EVT_THRESHOLD = evt_model['threshold']
EVT_SHAPE = evt_model['shape']
EVT_SCALE = evt_model['scale']
C_IDLE = 1.0
C_COLD = 10.0
CVAR_ALPHA = 0.95
CVAR_DELTA = 20.0
LAMBDA_PENALTY = 100.0
MAX_PREWARM_COUNT = 500
NUM_SCENARIOS = 1000
LOOKBACK_PERIOD = 10

# --- 2. THE BULLETPROOF HYBRID PREDICTOR FUNCTION ---
def predict_distribution(history):
    history = np.array(history).reshape((1, LOOKBACK_PERIOD, 1))
    tcn_predictions = tcn_model.predict(history, verbose=0)[0]
    
    p50_pred, p90_pred, p99_pred = tcn_predictions[0], tcn_predictions[1], tcn_predictions[2]
    
    if p99_pred < EVT_THRESHOLD:
        mode = max(p50_pred, p90_pred)
        scenarios = np.random.triangular(p50_pred, mode, p99_pred, size=NUM_SCENARIOS)
    else:
        right_bound = EVT_THRESHOLD
        mode_bound = min(p90_pred, right_bound)
        left_bound = min(p50_pred, mode_bound)

        if left_bound >= right_bound:
            extreme_exceedances = genpareto.rvs(c=EVT_SHAPE, scale=EVT_SCALE, size=NUM_SCENARIOS)
            scenarios = EVT_THRESHOLD + extreme_exceedances
        else:
            base_scenarios = np.random.triangular(left_bound, mode_bound, right_bound, size=int(NUM_SCENARIOS * CVAR_ALPHA))
            num_extreme = NUM_SCENARIOS - len(base_scenarios)
            extreme_exceedances = genpareto.rvs(c=EVT_SHAPE, scale=EVT_SCALE, size=num_extreme)
            extreme_scenarios = EVT_THRESHOLD + extreme_exceedances
            scenarios = np.concatenate([base_scenarios, extreme_scenarios])
            
    return np.maximum(0, scenarios).astype(int), tcn_predictions

# --- 3. THE ROBUST CVAR OPTIMIZER FUNCTION ---
def cvar_optimizer(scenarios):
    feasible_solutions = {}
    for c in range(MAX_PREWARM_COUNT + 1):
        overflows = np.maximum(0, scenarios - c)
        worst_overflows = np.sort(overflows)[int(len(overflows) * CVAR_ALPHA):]
        cvar = worst_overflows.mean() if len(worst_overflows) > 0 else 0
        
        if cvar <= CVAR_DELTA:
            total_cost = (C_IDLE * c) + (C_COLD * overflows.mean())
            feasible_solutions[c] = total_cost
    
    if feasible_solutions:
        best_c = min(feasible_solutions, key=feasible_solutions.get)
        return best_c

    # Fallback optimizer
    best_objective_value = float('inf')
    best_c_fallback = 0
    for c in range(MAX_PREWARM_COUNT + 1):
        overflows = np.maximum(0, scenarios - c)
        worst_overflows = np.sort(overflows)[int(len(overflows) * CVAR_ALPHA):]
        cvar = worst_overflows.mean() if len(worst_overflows) > 0 else 0
        penalty = LAMBDA_PENALTY * max(0, cvar - CVAR_DELTA)
        objective_value = (C_IDLE * c) + (C_COLD * overflows.mean()) + penalty
        
        if objective_value < best_objective_value:
            best_objective_value = objective_value
            best_c_fallback = c
            
    return best_c_fallback

# --- 4. SIMULATION AND PLOTTING ---
def run_simulation_and_generate_plots():
    print("\n--- Starting Simulation on Test Data ---")
    
    test_data_files = [os.path.join(f'R{i}_preprocessed', 'test_data.csv') for i in range(1, 6)]
    existing_files = [f for f in test_data_files if os.path.exists(f)]
    if not existing_files:
        print("Error: No test data found. Halting.")
        return
        
    test_df = pd.concat([pd.read_csv(f) for f in existing_files], ignore_index=True)
    test_df.sort_values(by='timestamp', inplace=True)
    
    X_test = test_df.filter(like='lag').values
    y_test = test_df['arrival_rate'].values
    
    results = []
    
    for i in range(len(X_test)):
        if i % 500 == 0: # Print progress less often to keep the log clean
            print(f"Simulating step {i}/{len(X_test)}...")
        
        history = X_test[i]
        actual_demand = y_test[i]
        
        scenarios, tcn_preds = predict_distribution(history)
        decision_ours = cvar_optimizer(scenarios)
        
        decision_reactive = int(history[-1])
        decision_fixed = int(np.median(y_test))
        decision_tcn_only = int(tcn_preds[1])
        
        models = {
            'Our Model (TCN+EVT+CVaR)': decision_ours,
            'TCN-Only (P90 Rule)': decision_tcn_only,
            'Reactive Scaler': decision_reactive,
            'Fixed Pool': decision_fixed
        }
        
        for name, decision in models.items():
            cold_starts = max(0, actual_demand - decision)
            total_cost = (max(0, decision - actual_demand) * C_IDLE) + (cold_starts * C_COLD)
            
            # ** THIS IS THE CORRECTED LINE **
            # We now correctly save the 'decision' for each step
            results.append({
                'model': name,
                'decision': decision,
                'cold_starts': cold_starts,
                'total_cost': total_cost
            })

    results_df = pd.DataFrame(results)
    print("Simulation complete. Generating graphs...")

    # --- Plotting ---
    sns.set_style("whitegrid")
    
    perf_data = results_df.groupby('model').agg(
        total_cost=('total_cost', 'sum'),
        total_cold_starts=('cold_starts', 'sum')
    ).reset_index()
    
    print("\n\n--- Final Numerical Results Summary ---")
    pd.set_option('display.float_format', '{:,.2f}'.format)
    print(perf_data)
    print("--------------------------------------\n")

    # Graph 1
    fig, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(data=perf_data, x='model', y='total_cost', ax=ax1, hue='model', palette='viridis', legend=False)
    ax1.set_ylabel('Total Cost ($)', color='b'); ax1.tick_params(axis='x', rotation=15)
    ax1.set_title('Performance Comparison vs. Baselines', fontsize=16)
    ax2 = ax1.twinx()
    sns.lineplot(data=perf_data, x='model', y='total_cold_starts', ax=ax2, color='r', marker='o', sort=False)
    ax2.set_ylabel('Total Cold Starts (SLA Violations)', color='r')
    plt.tight_layout(); plt.savefig('graph_1_performance_comparison.png')
    print("Saved graph 1: Performance Comparison")

    # Graph 2
    ablation_models = ['TCN-Only (P90 Rule)', 'Our Model (TCN+EVT+CVaR)']
    ablation_data = results_df[results_df['model'].isin(ablation_models)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ablation_data, x='model', y='total_cost', estimator=sum, palette='plasma', hue='model', legend=False, errorbar=None)
    plt.title('Ablation Study: Impact of EVT and CVaR Optimizer', fontsize=16)
    plt.ylabel('Total Cost ($)'); plt.xlabel('')
    plt.savefig('graph_2_ablation_study.png'); print("Saved graph 2: Ablation Study")

    # Graph 3
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=perf_data, x='total_cost', y='total_cold_starts', hue='model', s=200, palette='magma')
    plt.title('Cost vs. Performance Trade-off', fontsize=16)
    plt.xlabel('Total Cost (Lower is Better)'); plt.ylabel('Total Cold Starts (Lower is Better)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2); plt.tight_layout()
    plt.savefig('graph_3_cost_performance_tradeoff.png'); print("Saved graph 3: Cost vs. Performance Trade-off")
    
    # Graph 4
    slice_len = 200
    our_model_decisions = results_df[results_df['model'] == 'Our Model (TCN+EVT+CVaR)']['decision'].values[:slice_len]
    actuals = test_df['arrival_rate'].values[:slice_len]
    timesteps = np.arange(len(actuals))
    
    plt.figure(figsize=(15, 6))
    plt.plot(timesteps, actuals, label='Actual Demand', color='black', linewidth=2)
    plt.plot(timesteps, our_model_decisions, label='Our Model\'s Pre-warm Decision', color='crimson', linestyle='--')
    plt.fill_between(timesteps, actuals, our_model_decisions, where=our_model_decisions > actuals, color='yellow', alpha=0.3, label='Wasted Capacity')
    plt.fill_between(timesteps, actuals, our_model_decisions, where=actuals > our_model_decisions, color='red', alpha=0.3, label='Cold Starts (Deficit)')
    plt.title('Example Forecast vs. Actual Workload', fontsize=16)
    plt.xlabel('Time (Minutes)'); plt.ylabel('Number of Concurrent Users')
    plt.legend(); plt.savefig('graph_4_timeseries_example.png'); print("Saved graph 4: Time-Series Example")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    run_simulation_and_generate_plots()
    print("\nAll simulations and plotting are complete.")