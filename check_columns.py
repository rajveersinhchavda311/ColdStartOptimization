import pandas as pd
import os
import glob

# --- Find one of the CSV files to inspect ---
# We'll look inside the R1 folder
file_path = glob.glob(os.path.join('R1', '*.csv'))[0]

# Load just the first 5 rows to see the columns
df = pd.read_csv(file_path, nrows=5)

# Print the column names
print("The column names are:")
print(df.columns)