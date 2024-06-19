import pandas as pd
import os

def load_data(path):
    # Check if the file exists
    if not os.path.exists(path):
        print(f"Error: The file at path {path} does not exist.")
        return None

    # Try to load the CSV file
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Example usage
dataframe = load_data("data/raw/all_season_details.csv")
if dataframe is not None:
    print(dataframe.head())
