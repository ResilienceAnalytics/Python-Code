import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import sys

def apply_transformation(file_path, columns, method):
    """
    Perform Box-Cox or Yeo-Johnson transformation on the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to apply the transformation.
    - method: str, type of transformation ('boxcox' or 'yeojohnson').
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Select columns for transformation
    if columns != 'all':
        cols_to_transform = columns
    else:
        cols_to_transform = data.select_dtypes(include=[np.number]).columns

    # Apply transformation
    if method == 'boxcox':
        for col in cols_to_transform:
            # Adding an offset to handle non-positive values
            data[col], _ = stats.boxcox(data[col] + 1)
    elif method == 'yeojohnson':
        pt = PowerTransformer(method='yeo-johnson')
        data[cols_to_transform] = pt.fit_transform(data[cols_to_transform])
    else:
        raise ValueError("Invalid method. Choose 'boxcox' or 'yeojohnson'.")

    return data

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python transform_data.py <path_to_data_file> <columns/all> <method>")
        sys.exit(1)

    file_path, columns, method = sys.argv[1], sys.argv[2], sys.argv[3]
    if columns != 'all':
        columns = columns.split(',')
    transformed_data = apply_transformation(file_path, columns, method)
    
    print(transformed_data)
