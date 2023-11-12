import pandas as pd
import numpy as np
import sys

def logarithmic_transformation(file_path, columns):
    """
    Perform logarithmic transformation on the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to apply the logarithmic transformation.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Select columns for transformation
    if columns != 'all':
        cols_to_transform = columns
    else:
        cols_to_transform = data.select_dtypes(include=[np.number]).columns

    # Apply logarithmic transformation
    for col in cols_to_transform:
        # Avoiding logarithm of zero or negative numbers by adding an offset (1)
        data[col] = np.log(data[col] + 1)

    return data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python log_transform.py <path_to_data_file> <columns/all>")
        sys.exit(1)

    file_path, columns = sys.argv[1], sys.argv[2]
    if columns != 'all':
        columns = columns.split(',')
    transformed_data = logarithmic_transformation(file_path, columns)
    
    print(transformed_data)
