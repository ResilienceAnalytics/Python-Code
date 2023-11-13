import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import os
import sys

def read_file(file_path):
    _, file_ext = os.path.splitext(file_path)
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_ext == '.ods':
        return pd.read_excel(file_path, engine='odf')
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

def plot_histograms(numeric_data):
    root = tk.Tk()
    root.title("Data Distribution Histograms")

    tab_control = ttk.Notebook(root)
    num_cols = len(numeric_data.columns)
    num_figures = (num_cols + 3) // 4

    for i in range(num_figures):
        tab = ttk.Frame(tab_control)
        tab_control.add(tab, text=f'Figure {i + 1}')
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

        for j in range(4):
            idx = i * 4 + j
            if idx < num_cols:
                ax = axes[j // 2, j % 2]
                column = numeric_data.columns[idx]
                ax.hist(numeric_data[column], bins=50)
                ax.set_title(f'Distribution of {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
            else:
                axes[j // 2, j % 2].axis('off')

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    tab_control.pack(expand=1, fill='both')
    root.mainloop()

# Lecture du chemin du fichier depuis la ligne de commande
if len(sys.argv) < 2:
    print("Usage: python distribution.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]
df = read_file(file_path)
numeric_data = df.select_dtypes(include=['float64', 'int64'])

plot_histograms(numeric_data)
