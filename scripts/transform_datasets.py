import pandas as pd
import numpy as np
import os

def transform_dataset(filepath):
    """
    Enhances dataset by:
    - Addressing missing values.
    - Converting categorical features.
    - Scaling numerical attributes.

    Parameters:
        filepath (str): Path to the structured CSV file.

    Returns:
        pd.DataFrame: Enhanced dataset.
    """
    # Load the dataset
    dataset = pd.read_csv(filepath, delimiter=",", low_memory=False)

    # Classify columns into numerical and categorical
    numeric_features = dataset.select_dtypes(include=[np.number]).columns
    categorical_features = dataset.select_dtypes(exclude=[np.number]).columns

    print(f"📂 Transforming file: {filepath}")
    print(f"🧮 Numeric Features: {list(numeric_features)}")
    print(f"🔤 Categorical Features: {list(categorical_features)}\n")

    # Handle missing values efficiently
    for column in numeric_features:
        dataset[column] = dataset[column].fillna(dataset[column].median())  # Use median for missing numerics

    for column in categorical_features:
        if not dataset[column].mode().empty:
            dataset[column] = dataset[column].fillna(dataset[column].mode()[0])  # Replace missing categoricals with mode
        else:
            print(f"⚠️ Notice: No mode found for {column}, keeping NaN values.")

    # Scale numerical attributes using Min-Max Scaling
    for column in numeric_features:
        dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())

    return dataset

# Define directory locations
source_directory = "data/refined"
destination_directory = "data/transformed"
os.makedirs(destination_directory, exist_ok=True)

# Process and store transformed datasets
for file in os.listdir(source_directory):
    file_path = os.path.join(source_directory, file)
    
    if file.endswith(".csv"):
        try:
            enhanced_data = transform_dataset(file_path)
            output_file_path = os.path.join(destination_directory, f"transformed_{file}")
            enhanced_data.to_csv(output_file_path, index=False, sep=",")
            print(f"✅ File stored successfully: {output_file_path}\n")
        except Exception as err:
            print(f"❌ Issue transforming {file}: {err}\n")

print("🚀 Data transformation complete. All files saved!")
