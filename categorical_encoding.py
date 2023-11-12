import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def categorical_encoding(file_path, columns, encoding_type):
    """
    Perform categorical encoding on the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to apply the encoding.
    - encoding_type: str, type of encoding ('onehot' or 'label').
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Select columns for encoding
    if columns != 'all':
        cols_to_encode = columns
    else:
        cols_to_encode = data.select_dtypes(include=['object', 'category']).columns

    # Apply encoding
    if encoding_type == 'onehot':
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_data = encoder.fit_transform(data[cols_to_encode])
        # Create a DataFrame with encoded columns
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(cols_to_encode))
        data = pd.concat([data.drop(cols_to_encode, axis=1), encoded_df], axis=1)
    elif encoding_type == 'label':
        encoder = LabelEncoder()
        for col in cols_to_encode:
            data[col] = encoder.fit_transform(data[col])
    else:
        raise ValueError("Invalid encoding type. Choose 'onehot' or 'label'.")

    return data

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python categorical_encoding.py <path_to_data_file> <columns/all> <encoding_type>")
        sys.exit(1)

    file_path, columns, encoding_type = sys.argv[1], sys.argv[2], sys.argv[3]
    if columns != 'all':
        columns = columns.split(',')
    encoded_data = categorical_encoding(file_path, columns, encoding_type)
    
    print(encoded_data)
