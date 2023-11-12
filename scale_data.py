import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import sys

def scale_data(file_path, method, columns):
    """
    Scale the data in the file based on the chosen method.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - method: str, scaling method ('normalization', 'standardization', 'z-score').
    - columns: list of str, columns to scale. If 'all', scale all numeric columns.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Exclude 'DATE' column from scaling
    if 'DATE' in data.columns:
        date_data = data['DATE']
        data.drop('DATE', axis=1, inplace=True)
    else:
        date_data = None

    # Select columns to scale
    if columns != 'all':
        data = data[columns]

    # Choose the scaling method
    if method == 'normalization':
        scaler = MinMaxScaler()
    elif method in ['standardization', 'z-score']:
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Choose 'normalization', 'standardization', or 'z-score'.")

    # Scale the data
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))

    # Convert back to DataFrame and keep non-numeric columns
    scaled_data = pd.DataFrame(scaled_data, columns=data.select_dtypes(include=[np.number]).columns)
    non_numeric_data = data.select_dtypes(exclude=[np.number])
    scaled_data = pd.concat([non_numeric_data, scaled_data], axis=1)

    # Add back 'DATE' column if it was present
    if date_data is not None:
        scaled_data['DATE'] = date_data

    return scaled_data

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scale_data.py <path_to_data_file> <method> <columns/all>")
        sys.exit(1)

    file_path, method, columns = sys.argv[1], sys.argv[2], sys.argv[3]
    if columns != 'all':
        columns = columns.split(',')
    scaled_data = scale_data(file_path, method, columns)
    print(scaled_data)
