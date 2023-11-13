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
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path, engine='odf')

    # Select columns for analysis
    if columns != 'all':
        data = data[columns]

    # Compute descriptive statistics
    # Include datetime_is_numeric=True to treat datetime columns numerically
    desc_stats = data.describe(include='all', datetime_is_numeric=True)

    return desc_stats

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python descriptive_stats.py <path_to_data_file> <columns/all>")
        sys.exit(1)

    file_path, columns = sys.argv[1], sys.argv[2]
    if columns != 'all':
        columns = columns.split(',')
    stats = descriptive_statistics(file_path, columns)
    
    # Print the statistics to the terminal (will be redirected to a file if command includes redirection)
    print(stats)
