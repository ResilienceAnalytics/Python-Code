import pandas as pd
import sys
import os

def split_data_by_date(file_path, start_date, end_date, columns='all'):
    """
    Split data based on date range and save to a new file.

    Parameters:
    - file_path: str, path to the data file (CSV, XLSX, ODS, etc.).
    - start_date: str, start date in 'YYYY-MM-DD' format.
    - end_date: str, end date in 'YYYY-MM-DD' format.
    - columns: list or str, columns to include in the split ('all' for all columns).
    """
    # Determine file type and read the file accordingly
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.ods'):
        data = pd.read_excel(file_path, engine='odf')
    else:
        raise ValueError("Unsupported file format")

    # Convert 'DATE' column to datetime
    data['DATE'] = pd.to_datetime(data['DATE'])

    # Filter data based on date range
    mask = (data['DATE'] >= start_date) & (data['DATE'] <= end_date)
    filtered_data = data.loc[mask]

    # Select specified columns
    if columns != 'all':
        filtered_data = filtered_data[columns]

    # Save the filtered data to a new file
    output_file_path = f"filtered_{os.path.basename(file_path)}"
    if file_path.endswith('.csv'):
        filtered_data.to_csv(output_file_path, index=False)
    elif file_path.endswith('.xlsx'):
        filtered_data.to_excel(output_file_path, index=False)
    elif file_path.endswith('.ods'):
        filtered_data.to_excel(output_file_path, index=False, engine='odf')

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python split_data.py <path_to_data_file> <start_date> <end_date> [<columns>/all]")
        sys.exit(1)

    file_path = sys.argv[1]
    start_date, end_date = sys.argv[2], sys.argv[3]
    columns = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] != 'all' else 'all'
    
    split_data_by_date(file_path, start_date, end_date, columns)
