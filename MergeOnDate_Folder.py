import pandas as pd
import sys
import os

def merge_files_on_date(input_dir, output_file_path):
    """
    Merges multiple files based on the 'DATE' column within an input directory and saves the combined file.

    Parameters:
    - input_dir: str, directory containing the files to be merged.
    - output_file_path: str, path to save the combined file.
    """
    dataframes = []
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        # Make sure to only process files with the expected format
        if os.path.isfile(file_path) and file_path.endswith('.ods'):
            df = pd.read_excel(file_path, engine='odf')
            df['DATE'] = pd.to_datetime(df['DATE'])  # Ensure 'DATE' is in datetime format
            dataframes.append(df)
    
    if not dataframes:
        raise ValueError(f"No ODS files found in the directory: {input_dir}")

    # Merge all DataFrames on the 'DATE' column
    combined_data = pd.concat(dataframes).sort_values(by='DATE')

    # Save the combined data
    combined_data.to_excel(output_file_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_files_on_date.py <path_to_input_directory> <path_to_output_file>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file_path = sys.argv[2]

    merge_files_on_date(input_dir, output_file_path)
