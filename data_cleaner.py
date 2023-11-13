import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

# Function to load data based on file extension
def load_data(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        return pd.read_excel(file_path)
    elif file_extension == '.ods':
        return pd.read_excel(file_path, engine='odf')
    else:
        raise ValueError("Unsupported file format")

# Function to handle missing values
def handle_missing_values(data, method='delete', output_files=None):
    # Replace non-numeric placeholders with NaN
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Exclude columns with datetime objects
    non_datetime_data = data.select_dtypes(exclude=['datetime'])

    # Apply imputation on non-datetime data
    if method == 'delete':
        data_cleaned = non_datetime_data.dropna()
    elif method in ['impute_mean', 'impute_median', 'impute_mode']:
        strategy = {
            'impute_mean': 'mean',
            'impute_median': 'median',
            'impute_mode': 'most_frequent'
        }[method]

        imputer = SimpleImputer(strategy=strategy)
        non_datetime_data_cleaned = pd.DataFrame(imputer.fit_transform(non_datetime_data), columns=non_datetime_data.columns)

        # Combine the imputed non-datetime data with the datetime data
        datetime_data = data.select_dtypes(include=['datetime'])
        data_cleaned = pd.concat([datetime_data.reset_index(drop=True), non_datetime_data_cleaned.reset_index(drop=True)], axis=1)
    else:
        raise ValueError("Unsupported missing value handling method")

    # Save data if output_files are provided
    if output_files:
        for file_name, df in output_files.items():
            # Replace non-numeric placeholders with NaN for each output file
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Combine the imputed non-datetime data with the datetime data for each file
            datetime_data = data.select_dtypes(include=['datetime'])
            non_datetime_data_cleaned = df.select_dtypes(exclude=['datetime'])
            combined_data = pd.concat([datetime_data.reset_index(drop=True), non_datetime_data_cleaned.reset_index(drop=True)], axis=1)
            combined_data.to_csv(file_name, index=False)

    return data_cleaned

# Main script
file_path = input("Enter the path to the data file (CSV, XLSX, ODS): ")
data = load_data(file_path)

print("Choose how to handle missing values:")
print("1. Delete rows with missing values")
print("2. Impute missing values with mean")
print("3. Impute missing values with median")
print("4. Impute missing values with mode")
print("5. Hybrid method with multiple output files")

choice = input("Enter your choice (1-5): ")
methods = ['delete', 'impute_mean', 'impute_median', 'impute_mode']

if choice in ['1', '2', '3', '4']:
    method = methods[int(choice) - 1]
    data_cleaned = handle_missing_values(data, method=method)
    save = input("Do you want to save the cleaned data? (yes/no): ")
    if save.lower() == 'yes':
        output_dir = input("Enter the output directory path: ")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        output_file = f"cleaned_data_{method}.csv"
        output_path = os.path.join(output_dir, output_file)
        data_cleaned.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    else:
        print(data_cleaned)
elif choice == '5':
    output_dir = input("Enter the output directory path: ")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_files = {}
    for method in methods:
        file_name = f"{method}.csv"
        output_files[file_name] = handle_missing_values(data, method=method)
    data_cleaned = handle_missing_values(data, method='delete', output_files=output_files)
    for file_name, df in output_files.items():
        output_path = os.path.join(output_dir, file_name)
        df.to_csv(output_path, index=False)
        print(f"{file_name} saved to {output_path}")
else:
    print("Invalid choice")
