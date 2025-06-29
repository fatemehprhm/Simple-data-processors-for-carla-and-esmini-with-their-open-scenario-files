import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd


def adaptive_gaussian_filter(data, sigma, edge_threshold=0.1):
    if sigma == 0:
        return data
    filtered = gaussian_filter(data, sigma)
    gradient = np.abs(np.gradient(data))
    edge_mask = gradient > (edge_threshold * np.max(gradient))
    result = np.where(edge_mask, data, filtered)
    return result

def process_multiple_columns(input_path, params, output_path):
    df = pd.read_csv(input_path)
    filtered_data = {}
    
    n_cols = len(params)
    fig, axs = plt.subplots(n_cols, 1, figsize=(12, 6*n_cols), sharex=True)
    if n_cols == 1:
        axs = [axs]
    
    for i, (column_name, sigma, edge_threshold) in enumerate(params):
        data = df[column_name].values
        filtered = adaptive_gaussian_filter(data, sigma, edge_threshold)
        filtered_data[f'filtered_{column_name}'] = filtered
        
        axs[i].plot(df['x'], data, label='Original', alpha=0.5)
        axs[i].plot(df['x'], filtered, label='Filtered', linewidth=2)
        axs[i].set_title(f'{column_name} (σ={sigma}, edge_threshold={edge_threshold})')
        axs[i].set_ylabel(column_name)
        axs[i].legend()
        axs[i].grid(True)
    
    axs[-1].set_xlabel('time')
    plt.tight_layout()
    plt.show()
    
    for col, filtered in filtered_data.items():
        df[col] = filtered
    df.to_csv(output_path, index=False)
    
    return filtered_data

# Parameters: (column_name, sigma, edge_threshold)
params = [
    ('lateral_distance', 3, 1),
    ('d', 3, 1),
    ('a_alks', 3, 1),
    ('lateral_movement', 3, 1),
    ('d_lateral_distance', 3, 1),
    ('ttc', 0, 0)
]

input_path = 'setlabs_data/pr_output.csv'
output_path = 'setlabs_data/gaus_filter_output.csv'
filtered_data = process_multiple_columns(input_path, params, output_path)

print(f"Filtered data has been saved to {output_path}")