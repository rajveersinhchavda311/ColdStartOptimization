import dask.dataframe as dd
import pandas as pd
import os
import glob

# --- 1. CONFIGURATION ---
print("Starting the final, corrected preprocessing script for the Huawei dataset...")

REGION_FOLDERS = ['R1', 'R2', 'R3', 'R4', 'R5']
WINDOW_SIZE = '60s' 
LOOKBACK_PERIOD = 10

# --- 2. MAIN PROCESSING LOOP ---
for region in REGION_FOLDERS:
    print(f"\n=================================================")
    print(f"Processing data for Region: {region}")
    print(f"=================================================")

    input_path = os.path.join(os.getcwd(), region)
    output_path = os.path.join(os.getcwd(), f"{region}_preprocessed")

    os.makedirs(output_path, exist_ok=True)
    print(f"Ensured output directory exists: {output_path}")

    all_csv_files = glob.glob(os.path.join(input_path, '*.csv'))

    if not all_csv_files:
        print(f"Warning: No CSV files found in {input_path}. Skipping this region.")
        continue

    print(f"Found {len(all_csv_files)} CSV files to load for {region}.")
    
    # Specify dtype for 'time' column to avoid mixed-type warnings
    dask_df = dd.read_csv(all_csv_files, dtype={'time': 'float64'})
    print("Dask is ready to process the files.")

    # --- Step C: Create a Real Timestamp (Mathematical Method) ---
    # ** THIS IS THE FINAL, CORRECTED SECTION **
    
    # 1. Define a starting date for our dataset.
    base_date = pd.to_datetime('2025-09-01')

    # 2. Correct the 0-indexed 'day' by adding 1, then convert to a 'day' duration.
    day_offset = dd.to_timedelta(dask_df['day'], unit='D')
    
    # 3. Convert the 'time' column (in seconds) to a 'second' duration.
    time_offset = dd.to_timedelta(dask_df['time'], unit='s')
    
    # 4. Create the final, correct timestamp by adding the offsets to the base date.
    dask_df['timestamp'] = base_date + day_offset + time_offset
    
    # Set the new 'timestamp' column as the index
    dask_df = dask_df.set_index('timestamp')
    print("Timestamps created successfully using mathematical method.")

    # --- Step D: Calculate Arrival Rate in Time Windows ---
    arrival_rate_dask = dask_df.resample(WINDOW_SIZE).size()
    
    # --- Step E: The COMPUTE Step ---
    print("Executing computation with Dask... This will take a while.")
    arrival_rate_series = arrival_rate_dask.compute()
    pandas_df = arrival_rate_series.to_frame(name='arrival_rate')
    print(f"Computation for {region} complete! Result has {len(pandas_df)} rows.")

    # --- Step F: Feature Engineering (Creating Lag Features) ---
    print("Creating lag features...")
    for i in range(1, LOOKBACK_PERIOD + 1):
        pandas_df[f'lag_{i}'] = pandas_df['arrival_rate'].shift(i)
    
    pandas_df.dropna(inplace=True)
    print("Lag features created.")

    # --- Step G: Chronological Split ---
    print("Splitting data into train, validation, and test sets...")
    train_size = 0.6
    val_size = 0.2
    n = len(pandas_df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = pandas_df.iloc[:train_end]
    val_df = pandas_df.iloc[train_end:val_end]
    test_df = pandas_df.iloc[val_end:]
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows.")

    # --- Step H: Save the Processed Files ---
    print(f"Saving processed files to {output_path}...")
    train_df.to_csv(os.path.join(output_path, 'train_data.csv'))
    val_df.to_csv(os.path.join(output_path, 'val_data.csv'))
    test_df.to_csv(os.path.join(output_path, 'test_data.csv'))
    print(f"Successfully saved all files for region {region}.")

print("\n\nAll regions have been processed. Preprocessing is complete!")