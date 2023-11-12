import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import KBinsDiscretizer

def discretize_continuous_variables(file_path, columns, n_bins, strategy):
    """
    Perform discretization of continuous variables in the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to discretize.
    - n_bins: int, number of bins to use for discretization.
    - strategy: str, strategy for binning ('uniform', 'quantile', or 'kmeans').
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Select columns for discretization
    if columns != 'all':
        cols_to_discretize = columns
    else:
        cols_to_discretize = data.select_dtypes(include=[np.number]).columns

    # Apply discretization
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    data[cols_to_discretize] = discretizer.fit_transform(data[cols_to_discretize])

    return data

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python discretize_variables.py <path_to_data_file> <columns/all> <n_bins> <strategy>")
        sys.exit(1)

    file_path, columns, n_bins, strategy = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if columns != 'all':
        columns = columns.split(',')
    discretized_data = discretize_continuous_variables(file_path, columns, n_bins, strategy)
    
    print(discretized_data)
