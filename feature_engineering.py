import pandas as pd
import sys

def feature_engineering(file_path, new_features):
    """
    Perform feature engineering to create new derived variables.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - new_features: dict, dictionary with keys as new column names and values as expressions to generate these columns.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Create new features
    for new_col, expression in new_features.items():
        data[new_col] = eval(expression)

    return data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python feature_engineering.py <path_to_data_file> <new_features_dictionary>")
        sys.exit(1)

    file_path = sys.argv[1]
    new_features = eval(sys.argv[2])  # Convert string input to dictionary
    engineered_data = feature_engineering(file_path, new_features)
    
    print(engineered_data)
