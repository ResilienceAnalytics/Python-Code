import pandas as pd
import sys

def descriptive_statistics(file_path, columns):
    """
    Perform descriptive statistical analysis on the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to include in the analysis.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Select columns for analysis
    if columns != 'all':
        data = data[columns]

    # Compute descriptive statistics
    desc_stats = data.describe()

    return desc_stats

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python descriptive_stats.py <path_to_data_file> <columns/all>")
        sys.exit(1)

    file_path, columns = sys.argv[1], sys.argv[2]
    if columns != 'all':
        columns = columns.split(',')
    stats = descriptive_statistics(file_path, columns)
    
    print(stats)
