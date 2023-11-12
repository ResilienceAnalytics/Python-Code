import pandas as pd
import numpy as np
import sys

def compute_correlation(file_path, columns):
    """
    Compute the correlation matrix for the specified columns in the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to include in the correlation analysis.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Exclude 'DATE' column from analysis
    if 'DATE' in data.columns:
        data.drop('DATE', axis=1, inplace=True)

    # Select columns for correlation analysis
    if columns != 'all':
        data = data[columns]

    # Calculate the correlation matrix
    corr_matrix = data.corr(method='pearson')  # Default method is Pearson

    return corr_matrix

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python correlation_analysis.py <path_to_data_file> <columns/all>")
        sys.exit(1)

    file_path, columns = sys.argv[1], sys.argv[2]
    if columns != 'all':
        columns = columns.split(',')
    corr_matrix = compute_correlation(file_path, columns)
    print(corr_matrix)
