import sys
import os
import pandas as pd
import numpy as np

def load_and_convert_data(file_path):
    _, file_extension = os.path.splitext(file_path)
    data = None
    output_file_path = ''

    if file_extension.lower() in ['.csv', '.xlsx']:
        if file_extension.lower() == '.csv':
            data = pd.read_csv(file_path)
            output_file_path = file_path.replace('.csv', '.ods')
        elif file_extension.lower() == '.xlsx':
            data = pd.read_excel(file_path, engine='openpyxl')
            output_file_path = file_path.replace('.xlsx', '.ods')

        data.to_excel(output_file_path, engine='odf', index=False)
    elif file_extension.lower() == '.ods':
        data = pd.read_excel(file_path, engine='odf')
        output_file_path = file_path

    return data, output_file_path

def fill_missing_values(data, columns):
    if columns == 'all':
        columns = data.columns

    for column in columns:
        data[column] = data[column].replace({'.': np.nan, '': np.nan})
        data[column] = pd.to_numeric(data[column], errors='coerce')

        first_non_nan_index = data[column].first_valid_index()
        if first_non_nan_index is not None:
            data.loc[:first_non_nan_index, column] = data.loc[:first_non_nan_index, column].fillna(method='bfill')

        last_valid = None
        for i in range(len(data)):
            if not pd.isna(data.loc[i, column]):
                last_valid = data.loc[i, column]
            elif last_valid is not None:
                data.loc[i, column] = last_valid

    return data

def main(file_path, columns):
    data, converted_file_path = load_and_convert_data(file_path)
    data = fill_missing_values(data, columns)
    
    output_file_path = 'processed_data.ods'
    data.to_excel(output_file_path, engine='odf', index=False)

    return converted_file_path, output_file_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <file_path> <column_name> or <'all'>")
        sys.exit(1)

    file_path = sys.argv[1]
    columns = sys.argv[2:]

    if len(columns) == 1 and columns[0].lower() == 'all':
        columns = 'all'
    
    input_file, output_file = main(file_path, columns)
    print(f"Input file converted to ODS: {input_file}")
    print(f"Processed data saved to: {output_file}")
