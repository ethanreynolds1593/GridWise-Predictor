import pandas as pd
import os

def refine_dataset(filepath, date_columns=None, sep=";" ):
    """
    Refines a dataset by:
    - Converting specified or inferred datetime columns.
    - Handling missing values efficiently.
    - Ensuring correct data types for consistency.

    Parameters:
        filepath (str): Location of the CSV file.
        date_columns (list, optional): List of datetime columns. Auto-detects if None.
        sep (str): CSV delimiter.

    Returns:
        pd.DataFrame: Refined dataset.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(filepath, delimiter=sep, low_memory=False)

    # Identify datetime columns if not explicitly provided
    if date_columns is None:
        date_columns = [col for col in data.columns if "date" in col.lower() or "time" in col.lower()]

    # Convert detected datetime columns with error tolerance
    for column in date_columns:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    # Log missing datetime values
    missing_info = data[date_columns].isna().sum()
    print(f"ℹ️ Checking for missing values in datetime columns:\n{missing_info}")

    # Drop rows where all datetime columns are NaN
    if date_columns:
        data = data.dropna(subset=date_columns, how="all")

    # Warn if dataset is completely empty after cleaning
    if data.empty:
        print(f"⚠️ Warning: The dataset from {filepath} is empty after refining!")

    # Sort dataset by the first valid datetime column
    if date_columns and not data.empty:
        data = data.sort_values(by=date_columns[0]).reset_index(drop=True)

    return data

# Define directory paths
input_directory = "data/raw"
output_directory = "data/refined"
os.makedirs(output_directory, exist_ok=True)

# Iterate through all CSV files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        source_path = os.path.join(input_directory, filename)
        print(f"🔍 Now refining: {filename}...")

        # Refine and save the cleaned dataset
        refined_data = refine_dataset(source_path)
        
        if not refined_data.empty:  # Save only non-empty results
            output_path = os.path.join(output_directory, f"refined_{filename}")
            refined_data.to_csv(output_path, index=False, sep=",")
            print(f"✅ Successfully saved: {output_path}\n")
        else:
            print(f"⚠️ Skipping {filename} as the cleaned data is empty.\n")

print("🎯 All files have been successfully refined and saved!")
