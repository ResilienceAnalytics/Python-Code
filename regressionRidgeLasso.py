import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import sys

def perform_regression(file_path, columns, method, alpha):
    """
    Perform Ridge or Lasso regression on the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to include in the regression.
    - method: str, regression method ('ridge' or 'lasso').
    - alpha: float, regularization strength.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # Exclude 'DATE' column from analysis
    if 'DATE' in data.columns:
        data.drop('DATE', axis=1, inplace=True)

    # Select columns for regression
    if columns != 'all':
        data = data[columns]

    # Assume last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Perform regression
    if method == 'ridge':
        model = Ridge(alpha=alpha)
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        raise ValueError("Invalid method. Choose 'ridge' or 'lasso'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)

    return model.coef_, r2

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python regression.py <path_to_data_file> <columns/all> <method> <alpha>")
        sys.exit(1)

    file_path, columns, method, alpha = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    if columns != 'all':
        columns = columns.split(',')
    coefficients, r2_score = perform_regression(file_path, columns, method, alpha)
    
    print("Model Coefficients:\n", coefficients)
    print("\nR² Score:\n", r2_score)
