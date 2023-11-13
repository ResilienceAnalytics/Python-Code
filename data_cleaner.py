import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

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

def impute_data(data, strategy):
    # Conversion des colonnes 'object' en 'category' et remplacement des marqueurs spéciaux
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].replace({'.': np.nan}, inplace=True)
            data[col] = data[col].astype('category')

    # Séparation des types de données
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    categorical_data = data.select_dtypes(include=['category'])
    datetime_data = data.select_dtypes(include=['datetime64[ns]'])

    # Imputation pour les données numériques
    if not numerical_data.empty:
        num_imputer = SimpleImputer(strategy=strategy)
        numerical_data = pd.DataFrame(num_imputer.fit_transform(numerical_data), columns=numerical_data.columns)

    # Imputation pour les données catégorielles
    if not categorical_data.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        categorical_data = pd.DataFrame(cat_imputer.fit_transform(categorical_data), columns=categorical_data.columns)

    return pd.concat([datetime_data, numerical_data, categorical_data], axis=1)

# Entrée de l'utilisateur
file_path = input("Enter the path to the data file (CSV, XLSX, ODS): ")
data = load_data(file_path)

# Choix de l'imputation
print("Choose the imputation method for numerical data:")
print("1. Mean")
print("2. Median")
print("3. Hybrid (Mean and Median)")
imputation_choice = input("Enter your choice (1-3): ")
imputation_strategies = ['mean', 'median', 'hybrid']

if imputation_choice in ['1', '2']:
    strategy = imputation_strategies[int(imputation_choice) - 1]
    data_cleaned = impute_data(data, strategy=strategy)
elif imputation_choice == '3':
    # Imputation hybride
    data_cleaned_mean = impute_data(data, strategy='mean')
    data_cleaned_median = impute_data(data, strategy='median')
    strategy = 'hybrid'  # Définition de la variable 'strategy' pour l'option hybride
else:
    print("Invalid choice")
    exit()

# Sauvegarde des données nettoyées
save = input("Do you want to save the cleaned data? (yes/[Enter] for no): ").lower()
if save == '' or save == 'yes':
    _, file_name = os.path.split(file_path)
    base_name, _ = os.path.splitext(file_name)
    output_dir = base_name + "_" + strategy
    os.makedirs(output_dir, exist_ok=True)

    if strategy == 'hybrid':
        data_cleaned_mean.to_csv(os.path.join(output_dir, base_name + "_mean.csv"), index=False)
        data_cleaned_median.to_csv(os.path.join(output_dir, base_name + "_median.csv"), index=False)
        print(f"Data saved to {output_dir} with mean and median imputations")
    else:
        output_file = base_name + f"_{strategy}.csv"
        data_cleaned.to_csv(os.path.join(output_dir, output_file), index=False)
        print(f"Data saved to {os.path.join(output_dir, output_file)}")
elif save == 'no':
    if strategy == 'hybrid':
        print("Mean Imputation Data:")
        print(data_cleaned_mean)
        print("\nMedian Imputation Data:")
        print(data_cleaned_median)
    else:
        print(data_cleaned)
else:
    print("Invalid input for save option.")
    exit()
