import pandas as pd
import sys
import os

def convert_spreadsheet(input_file_path, output_dir, output_format):
    """
    Convert a spreadsheet from one format to another and save it to the output directory.

    Parameters:
    - input_file_path: str, path to the input file.
    - output_dir: str, directory to save the converted files.
    - output_format: str, desired output file format ('xlsx', 'csv', 'ods').
    """
    # Determine the file format from the extension
    file_extension = os.path.splitext(input_file_path)[1]
    if file_extension in ['.xls', '.xlsx']:
        data = pd.read_excel(input_file_path)
    elif file_extension == '.ods':
        data = pd.read_excel(input_file_path, engine='odf')
    elif file_extension == '.csv':
        data = pd.read_csv(input_file_path)  # Added this line to read .csv files
    else:
        raise ValueError("Unsupported input file format: " + file_extension)

    # Prepare the output file path
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file_path = os.path.join(output_dir, file_name + '.' + output_format)

    # Save the data in the desired output format
    if output_format == 'xlsx':
        data.to_excel(output_file_path, index=False)
    elif output_format == 'csv':
        data.to_csv(output_file_path, index=False)
    elif output_format == 'ods':
        data.to_excel(output_file_path, index=False, engine='odf')
    else:
        raise ValueError("Unsupported output file format: " + output_format)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_spreadsheet_folder.py <input_directory> <output_directory> <output_format>")
        sys.exit(1)

    input_dir, output_dir, output_format = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert all files in the input directory
    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(input_file_path):
            try:
                convert_spreadsheet(input_file_path, output_dir, output_format)
            except ValueError as e:
                print(f"Skipping {input_file_path}: {e}")
