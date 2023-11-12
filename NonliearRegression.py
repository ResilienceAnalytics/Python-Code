import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys

def perform_nonlinear_regression(file_path, columns, degree):
    """
    Perform nonlinear regression on the dataset.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - columns: list of str or 'all', columns to include in the regression.
    - degree: int, degree of the polynomial for the regression.
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

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

    # Perform regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)

    return model.coef_, r2

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python nonlinear_regression.py <path_to_data_file> <columns/all> <degree>")
        sys.exit(1)

    file_path, columns, degree = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if columns != 'all':
        columns = columns.split(',')
    coefficients, r2_score = perform_nonlinear_regression(file_path, columns, degree)
    
    print("Model Coefficients:\n", coefficients)
    print("\nR² Score:\n", r2_score)
