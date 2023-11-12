import pandas as pd
import sys
import os

def convert_spreadsheet(input_file_path, output_file_path, output_format):
    """
    Convert a spreadsheet from one format to another.

    Parameters:
    - input_file_path: str, path to the input file.
    - output_file_path: str, path to save the converted file.
    - output_format: str, desired output file format ('xlsx', 'csv', 'ods').
    """
    # Read the input file
    file_extension = os.path.splitext(input_file_path)[1]
    if file_extension in ['.xls', '.xlsx']:
        data = pd.read_excel(input_file_path)
    elif file_extension == '.ods':
        data = pd.read_excel(input_file_path, engine='odf')
    else:
        raise ValueError("Unsupported input file format: " + file_extension)

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
        print("Usage: python convert_spreadsheet.py <input_file_path> <output_file_path> <output_format>")
        sys.exit(1)

    input_file_path, output_file_path, output_format = sys.argv[1], sys.argv[2], sys.argv[3]
    convert_spreadsheet(input_file_path, output_file_path, output_format)
