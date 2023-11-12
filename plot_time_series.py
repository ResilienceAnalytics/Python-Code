import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_time_series(file_path, start_date, end_date, columns=None):
    """
    Plot time series data from the specified file within the given date range.
    If 'all' is specified for start_date and end_date, the entire date range in the dataset will be used.

    Parameters:
    - file_path: str, path to the data file (Excel or CSV).
    - start_date: str, start date for the plot (YYYY-MM-DD) or 'all'.
    - end_date: str, end date for the plot (YYYY-MM-DD) or 'all'.
    - columns: list of str or None, columns to plot. If None, plot all columns.
    """
    # Read the file
    data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)

    # Use entire date range if 'all' is specified
    if start_date.lower() == 'all' and end_date.lower() == 'all':
        filtered_data = data
    else:
        filtered_data = data.loc[start_date:end_date]

    # Select columns to plot
    if columns is not None:
        filtered_data = filtered_data[columns]

    # Plotting
    plt.figure(figsize=(10, 6))
    for col in filtered_data.columns:
        plt.plot(filtered_data.index, filtered_data[col], label=col)
    plt.title('Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_time_series.py <path_to_data_file> <start_date/all> <end_date/all> [<columns>]")
        sys.exit(1)

    file_path, start_date, end_date = sys.argv[1], sys.argv[2], sys.argv[3]
    columns = sys.argv[4].split(',') if len(sys.argv) > 4 else None
    plot_time_series(file_path, start_date, end_date, columns)
