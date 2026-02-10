import pandas as pd
import os
import glob

print("Running the debug script to check the timestamp format...")

try:
    # Find one of the CSV files to inspect (we only need one)
    # This looks for any CSV file inside the 'R1' folder
    file_path = glob.glob(os.path.join('R1', '*.csv'))[0]

    # Load just the first 5 rows to inspect the data quickly
    df = pd.read_csv(file_path, nrows=5)

    # Recreate the exact string that the main script tries to build
    df['datetime_str'] = '2025-09-' + df['day'].astype(str) + ' ' + df['time'].astype(str)

    print("\nHere are the first 5 date-time strings the script is trying to parse:")
    print("--------------------------------------------------------------------")
    # Print each generated string on a new line so we can see it clearly
    for item in df['datetime_str']:
        print(item)
    print("--------------------------------------------------------------------")
    print("\nPlease copy the lines between the ----- and paste them in your reply.")

except IndexError:
    print("\nError: Could not find any CSV files in the 'R1' folder.")
except Exception as e:
    print(f"\nAn error occurred: {e}")