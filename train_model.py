import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN
import os
import optuna

# --- 1. CONFIGURATION ---
print("Starting the TCN model training script with Optuna on ALL REGIONS...")

QUANTILES = [0.50, 0.90, 0.99]
LOOKBACK_PERIOD = 10
MODEL_SAVE_PATH = 'best_tcn_model_all_regions.keras'
# The number of different hyperparameter combinations Optuna will test
N_TRIALS = 10

# --- 2. DEFINE THE CUSTOM LOSS FUNCTION (PINBALL LOSS) ---
def pinball_loss(y_true, y_pred):
    loss = 0
    for i, q in enumerate(QUANTILES):
        error = y_true - y_pred[:, i]
        loss += tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss

# --- 3. LOAD AND PREPARE THE DATA (Updated to use all regions) ---
print("Loading and preparing data from ALL regions...")

REGION_FOLDERS = ['R1_preprocessed', 'R2_preprocessed', 'R3_preprocessed', 'R4_preprocessed', 'R5_preprocessed']
all_train_dfs = []
all_val_dfs = []

# Loop through each region's preprocessed folder
for region_folder in REGION_FOLDERS:
    print(f"Loading data from {region_folder}...")
    
    data_path = os.path.join(region_folder, 'train_data.csv')
    val_path = os.path.join(region_folder, 'val_data.csv')
    
    # Check if files exist before trying to load them
    if os.path.exists(data_path) and os.path.exists(val_path):
        # Load the training and validation data for the current region
        train_df_part = pd.read_csv(data_path)
        val_df_part = pd.read_csv(val_path)
        
        # Add the loaded data to our lists
        all_train_dfs.append(train_df_part)
        all_val_dfs.append(val_df_part)
    else:
        print(f"Warning: Data not found for {region_folder}. Skipping.")

# Combine the data from all regions into a single large DataFrame
train_df = pd.concat(all_train_dfs, ignore_index=True)
val_df = pd.concat(all_val_dfs, ignore_index=True)

print("\nAll regional data has been combined.")

# The rest of the preparation is the same
X_train = train_df.filter(like='lag').values
y_train = train_df['arrival_rate'].values
X_val = val_df.filter(like='lag').values
y_val = val_df['arrival_rate'].values

X_train = X_train.reshape((X_train.shape[0], LOOKBACK_PERIOD, 1))
X_val = X_val.reshape((X_val.shape[0], LOOKBACK_PERIOD, 1))
print(f"Data prepared: {X_train.shape[0]} total training samples.")


# --- 4. DEFINE THE OBJECTIVE FUNCTION FOR OPTUNA ---
def objective(trial):
    nb_filters = trial.suggest_int('nb_filters', 32, 128, step=16)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model = Sequential([
        TCN(input_shape=(LOOKBACK_PERIOD, 1),
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=[1, 2, 4, 8],
            use_skip_connections=True,
            dropout_rate=dropout_rate),
        Dense(len(QUANTILES))
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=pinball_loss)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    val_loss = min(history.history['val_loss'])
    return val_loss

# --- 5. RUN THE OPTUNA STUDY ---
print(f"Starting Optuna study with {N_TRIALS} trials...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

print("\nHyperparameter search complete!")
print(f"Best validation loss: {study.best_value}")
print("Best hyperparameters found:")
print(study.best_params)

# --- 6. RETRAIN AND SAVE THE BEST MODEL ---
print("\nRetraining the final model with the best hyperparameters...")
best_params = study.best_params

final_model = Sequential([
    TCN(input_shape=(LOOKBACK_PERIOD, 1),
        nb_filters=best_params['nb_filters'],
        kernel_size=best_params['kernel_size'],
        dilations=[1, 2, 4, 8],
        use_skip_connections=True,
        dropout_rate=best_params['dropout_rate']),
    Dense(len(QUANTILES))
])

final_optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
final_model.compile(optimizer=final_optimizer, loss=pinball_loss)

# For the final training, we can combine all data and train for a few epochs
# Or use the original training/validation sets with early stopping
final_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
    verbose=1
)

final_model.save(MODEL_SAVE_PATH)
print(f"Final, optimized model saved to {MODEL_SAVE_PATH}. Step 2 is complete!")