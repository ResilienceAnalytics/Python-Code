import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys

def feature_selection(file_path, columns, method, n_features):
    """
    Perform feature selection on the dataset using specified method.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to include in the feature selection.
    - method: str, method of feature selection ('rfe' or 'random_forest').
    - n_features: int, number of features to select.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Exclude 'DATE' column from analysis
    if 'DATE' in data.columns:
        data.drop('DATE', axis=1, inplace=True)

    # Select columns for feature selection
    if columns != 'all':
        data = data[columns]

    # Assume last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Feature selection method
    if method == 'rfe':
        estimator = LogisticRegression()
        selector = RFE(estimator, n_features_to_select=n_features)
    elif method == 'random_forest':
        estimator = RandomForestClassifier()
        selector = RFE(estimator, n_features_to_select=n_features)
    else:
        raise ValueError("Invalid method. Choose 'rfe' or 'random_forest'.")

    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_]

    return selected_features

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python feature_selection.py <path_to_data_file> <columns/all> <method> <n_features>")
        sys.exit(1)

    file_path, columns, method, n_features = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    if columns != 'all':
        columns = columns.split(',')
    selected_features = feature_selection(file_path, columns, method, n_features)
    
    print("Selected Features:\n", selected_features)
