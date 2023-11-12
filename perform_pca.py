import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

def perform_pca(file_path, columns, n_components):
    """
    Perform Principal Component Analysis on the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to include in the PCA analysis.
    - n_components: int, number of principal components to retain.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Exclude 'DATE' column from analysis
    if 'DATE' in data.columns:
        data.drop('DATE', axis=1, inplace=True)

    # Select columns for PCA analysis
    if columns != 'all':
        data = data[columns]

    # Standardize the data
    data_std = StandardScaler().fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_std)

    return principal_components, pca.explained_variance_ratio_

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python pca_analysis.py <path_to_data_file> <columns/all> <n_components>")
        sys.exit(1)

    file_path, columns, n_components = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if columns != 'all':
        columns = columns.split(',')
    principal_components, variance_ratio = perform_pca(file_path, columns, n_components)
    
    print("Principal Components:\n", principal_components)
    print("\nExplained Variance Ratio:\n", variance_ratio)
