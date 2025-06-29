import argparse
import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from scipy.signal import convolve
from scipy.signal import savgol_filter
from pathlib import Path


class processor(object):

    def final_process(self, df, i, results):
        # Initialize state and uncertainty
        data = df.iloc[:, i]
        column_name = df.columns[i]
        window_size = 50
        #moving_average = data.rolling(window=window_size).mean()
        #moving_average = moving_average.shift(-window_size // 2)

        # Create a symmetric Hamming window
        hamming_window = np.hamming(window_size)
        # Normalize the Hamming window
        hamming_window /= hamming_window.sum()
        # Calculate the half-window size
        half_window = (window_size) // 2
        # Pad the signal with zeros at the beginning and end
        padded_data = np.pad(data, (half_window, half_window), mode='edge')
        # Apply the centered moving average using convolution
        moving_average = np.convolve(padded_data, hamming_window, mode='valid')
        window_size = 10
        poly_order = 1
        smoothed_y = savgol_filter(moving_average, window_size, poly_order)
        results[column_name] = smoothed_y[:-1]

        plt.figure(figsize=(10, 6))
        plt.plot(df['time'].values,data.values,label='Input')
        plt.plot(df['time'].values,smoothed_y[:-1],color='r',label='output')
        plt.legend(fontsize=14)
        plt.xlabel('Time(sec)',fontsize=14)
        plt.ylabel(column_name,fontsize=14)
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Filtering data using moving average.")
    parser.add_argument("--csv", help="Input CSV file path")
    parser.add_argument("--output", help="Output CSV file name")
    parser.add_argument("-cn", "--column_num", type=int, help="Output CSV file name")
    parser.add_argument("--save", action='store_true', help="Save final info in yaml file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.drop(columns=df.columns[0])
    proc = processor()
    results = pd.DataFrame()
    proc.final_process(df, args.column_num, results)

    if args.save:
        file_path = args.output
        if Path(file_path).is_file():
            existing_data = pd.read_csv(args.output)
            updated_data = pd.concat([existing_data, results], axis=1)
            updated_data.to_csv(args.output, index=False)
        else:
            results.to_csv(args.output, index=False)
    print("Data filtered successfully.")
