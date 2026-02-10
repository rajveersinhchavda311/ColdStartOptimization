import pandas as pd
from scipy.stats import genpareto
import joblib
import os

print("Starting the EVT model building script on ALL REGIONS...")

# --- 1. CONFIGURATION ---
REGION_FOLDERS = ['R1_preprocessed', 'R2_preprocessed', 'R3_preprocessed', 'R4_preprocessed', 'R5_preprocessed']
EXTREME_THRESHOLD_PERCENTILE = 0.99
EVT_MODEL_SAVE_PATH = 'evt_model_all_regions.joblib'

# --- 2. LOAD AND COMBINE TRAINING DATA FROM ALL REGIONS ---
print("Loading and combining training data from all regions...")
all_train_dfs = []

for region_folder in REGION_FOLDERS:
    data_path = os.path.join(region_folder, 'train_data.csv')
    if os.path.exists(data_path):
        train_df_part = pd.read_csv(data_path)
        all_train_dfs.append(train_df_part)
    else:
        print(f"Warning: Training data not found for {region_folder}. Skipping.")

# Combine all the training data into a single DataFrame
train_df = pd.concat(all_train_dfs, ignore_index=True)
arrival_rates = train_df['arrival_rate']
print(f"Total of {len(arrival_rates)} training samples loaded.")

# --- 3. FIND THRESHOLD AND EXCEEDANCES ---
print(f"Calculating the {EXTREME_THRESHOLD_PERCENTILE*100}th percentile threshold...")
threshold = arrival_rates.quantile(EXTREME_THRESHOLD_PERCENTILE)
print(f"Extreme event threshold set at: {threshold:.2f}")

exceedances = arrival_rates[arrival_rates > threshold] - threshold
print(f"Found {len(exceedances)} exceedances to fit the EVT model.")

# --- 4. FIT THE GENERALIZED PARETO DISTRIBUTION (GPD) ---
print("Fitting the GPD model to the exceedances...")
# We fit the GPD to our exceedances.
# floc=0 fixes the location parameter for stability
shape, loc, scale = genpareto.fit(exceedances, floc=0)
print(f"GPD model fitted. Shape (xi): {shape:.4f}, Scale (sigma): {scale:.4f}")

# --- 5. SAVE THE EVT MODEL PARAMETERS ---
evt_model = {
    'threshold': threshold,
    'shape': shape,
    'scale': scale
}
joblib.dump(evt_model, EVT_MODEL_SAVE_PATH)
print(f"EVT model saved successfully to {EVT_MODEL_SAVE_PATH}. Step 3 is complete!")