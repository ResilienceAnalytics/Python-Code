import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

def main(file_path, variables, nan_strategy, start_date, end_date):
    """
    Perform linear regression analysis on a dataset.

    Parameters:
    - file_path: str, path to the input ODS file.
    - variables: str, independent and dependent variables separated by a pipe '|'.
    - nan_strategy: str, strategy for handling NaN values ('mean', 'median', or 'drop').
    - start_date: str, start date for filtering data in YYYY-MM-DD format.
    - end_date: str, end date for filtering data in YYYY-MM-DD format.
    """

    # Split the variables into independent and dependent variables
    independent_vars, dependent_var = variables.split(' | ')

    # Convert variable strings into lists
    independent_vars = independent_vars.split()  # Independent variables
    dependent_var = dependent_var.strip()  # Dependent variable

    # Validate NaN strategy
    if nan_strategy not in ['mean', 'median', 'drop']:
        print("Error: The strategy for handling NaNs must be 'mean', 'median', or 'drop'.")
        sys.exit(1)

    # Load the data from an ODS file
    data = pd.read_excel(file_path, engine='odf')

    # Filter the data between the start and end dates
    data['DATE'] = pd.to_datetime(data['DATE'])
    data = data[(data['DATE'] >= pd.to_datetime(start_date)) & (data['DATE'] <= pd.to_datetime(end_date))]

    # Handle NaN values based on the specified strategy
    if nan_strategy == 'drop':
        data.dropna(subset=independent_vars + [dependent_var], inplace=True)
    else:
        imputer = SimpleImputer(missing_values=np.nan, strategy=nan_strategy)
        data[independent_vars] = imputer.fit_transform(data[independent_vars])
        data[[dependent_var]] = imputer.fit_transform(data[[dependent_var]])

    # Select variables for regression
    X = data[independent_vars]  # Independent variables
    y = data[dependent_var].values.ravel()  # Dependent variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f'Coefficients: {model.coef_}')
    print(f'Interception: {model.intercept_}')
    print(f'Coefficient of determination (R²): {r2}')
    print(f'Mean Squared Error (MSE): {mse}')

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python regression.py <path_to_ods_file> 'DJIA SP500 NASDAQCOM | M1SL' mean 2019-12-01 2021-01-01")
        sys.exit(1)

    # Unpack command-line arguments
    _, file_path, variables, nan_strategy, start_date, end_date = sys.argv

    main(file_path, variables, nan_strategy, start_date, end_date)
