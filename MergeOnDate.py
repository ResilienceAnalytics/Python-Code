import pandas as pd
import sys

def merge_files_on_date(file_paths, output_file_path):
    """
    Merges multiple files based on the 'DATE' column and saves the combined file.

    Parameters:
    - file_paths: list of str, paths to the ODS files to be merged.
    - output_file_path: str, path to save the combined ODS file.
    """
    dataframes = []
    for file_path in file_paths:
        df = pd.read_excel(file_path, engine='odf')
        df['DATE'] = pd.to_datetime(df['DATE'])  # Ensure 'DATE' is in datetime format
        dataframes.append(df)

    # Merge all DataFrames on the 'DATE' column
    combined_data = pd.concat(dataframes).sort_values(by='DATE')

    # Save the combined data
    # Pandas doesn't support saving directly to ODS format, so we use Excel format.
    combined_data.to_excel(output_file_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python combine.py <path_to_output_ods_file> <path_to_first_ods_file> <path_to_second_ods_file> ...")
        sys.exit(1)

    # Unpack command-line arguments
    output_file_path = sys.argv[1]
    file_paths = sys.argv[2:]

    merge_files_on_date(file_paths, output_file_path)
