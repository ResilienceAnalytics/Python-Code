import os
import sys
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def load_and_convert_data(file_path):
    """
    Charge les données à partir du chemin de fichier spécifié et les convertit en format ODS si nécessaire.
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in ['.csv', '.xlsx']:
        if file_extension.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension.lower() == '.xlsx':
            data = pd.read_excel(file_path, engine='openpyxl')
        data['DATE'] = pd.to_datetime(data['DATE'])
        if file_extension.lower() != '.ods':
            output_file_path = file_path.replace(file_extension, '.ods')
            data.to_excel(output_file_path, engine='odf', index=False)
    elif file_extension.lower() == '.ods':
        data = pd.read_excel(file_path, engine='odf')
        data['DATE'] = pd.to_datetime(data['DATE'])
    else:
        raise ValueError("Unsupported file format.")
    return data

def segment_data(data, breakpoints):
    """
    Divise les données en segments basés sur les points de rupture.
    """
    segments = []
    for _, row in breakpoints.iterrows():
        start_date = row['Previous Breakpoint Date']
        end_date = row['Next Breakpoint Date']
        segment = data[(data['DATE'] >= start_date) & (data['DATE'] < end_date)].copy()
        segment.set_index('DATE', inplace=True)
        segments.append(segment)
    return segments

def fit_arima_to_segments(segments, data_column, order=(1, 1, 1)):
    """
    Ajuste un modèle ARIMA à chaque segment de données sur la colonne spécifiée.
    """
    models = []
    for segment in segments:
        model = ARIMA(segment[data_column], order=order)
        fitted_model = model.fit()
        models.append(fitted_model)
    return models

def plot_segments(segments, models):
    """
    Crée des graphiques pour visualiser les segments de données avant et après traitement par ARIMA.
    """
    num_segments = len(segments)
    fig, axs = plt.subplots(num_segments, 2, figsize=(15, 5 * num_segments))

    for i, (segment, model) in enumerate(zip(segments, models)):
        # Tracer les données originales
        axs[i, 0].plot(segment.index, segment['M1SL'], label='Données Originales')
        axs[i, 0].set_title(f'Segment {i+1} - Données Originales')
        axs[i, 0].legend()

        # Prévisions du modèle ARIMA
        forecast = model.predict(start=segment.index[0], end=segment.index[-1])
        if len(segment.index) == len(forecast):
            axs[i, 1].plot(segment.index, forecast, label='Prévisions ARIMA', color='orange')
        else:
            print(f"Erreur de dimension pour le segment {i+1}: données ({len(segment.index)}) vs prévisions ({len(forecast)})")
        axs[i, 1].set_title(f'Segment {i+1} - Prévisions ARIMA')
        axs[i, 1].legend()

    plt.tight_layout()
    plt.show()

def main(data_file, breakpoints_file, data_column):
    # Charger les données et les diviser en segments
    data = load_and_convert_data(data_file)
    breakpoints = load_and_convert_data(breakpoints_file)
    segments = segment_data(data, breakpoints)

    # Ajuster un modèle ARIMA à chaque segment
    models = fit_arima_to_segments(segments, data_column)

    # Afficher le résumé de chaque modèle ARIMA
    for i, model in enumerate(models):
        print(f"Modèle ARIMA pour le segment {i+1}:")
        print(model.summary())

    # Visualiser les graphiques des segments avant et après traitement ARIMA
    plot_segments(segments, models)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <data_file_path> <breakpoints_file_path> <data_column_name>")
        sys.exit(1)

    data_file = sys.argv[1]
    breakpoints_file = sys.argv[2]
    data_column = sys.argv[3]
    main(data_file, breakpoints_file, data_column)
