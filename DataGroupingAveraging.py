import pandas as pd
import os
import sys

def load_data(file_path):
    """
    Loads data from a CSV, ODS, or XLSX file.
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in ['.csv']:
        return pd.read_csv(file_path)
    elif file_extension.lower() in ['.xlsx']:
        return pd.read_excel(file_path, engine='openpyxl')
    elif file_extension.lower() in ['.ods']:
        return pd.read_excel(file_path, engine='odf')
    else:
        raise ValueError("Unsupported file format.")

def transform_data(data):
    """
    Transforms the data by grouping by date and calculating the average.
    """
    data['DATE'] = pd.to_datetime(data['DATE'])
    return data.groupby('DATE').mean().reset_index()

def main(input_file, output_file):
    data = load_data(input_file)
    transformed_data = transform_data(data)
    transformed_data.to_csv(output_file, index=False)
    print(f"Transformed data has been saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
