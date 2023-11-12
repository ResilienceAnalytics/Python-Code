import pandas as pd
import sys

def manage_temporal_data(file_path, columns):
    """
    Perform management of temporal data in the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', temporal columns to manage.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Select columns for temporal data management
    if columns != 'all':
        cols_to_manage = columns
    else:
        cols_to_manage = data.select_dtypes(include=['datetime64[ns]']).columns

    # Manage temporal data
    for col in cols_to_manage:
        data[col] = pd.to_datetime(data[col])
        data[f'{col}_year'] = data[col].dt.year
        data[f'{col}_month'] = data[col].dt.month
        data[f'{col}_day'] = data[col].dt.day
        data[f'{col}_hour'] = data[col].dt.hour
        data[f'{col}_minute'] = data[col].dt.minute
        data[f'{col}_second'] = data[col].dt.second

    return data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python temporal_data_management.py <path_to_data_file> <columns/all>")
        sys.exit(1)

    file_path, columns = sys.argv[1], sys.argv[2]
    if columns != 'all':
        columns = columns.split(',')
    managed_data = manage_temporal_data(file_path, columns)
    
    print(managed_data)
