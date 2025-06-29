import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = 1000  # Start with high uncertainty
        self.last_estimate = initial_value

    def update(self, measurement):
        # Prediction
        prediction = self.last_estimate
        prediction_error = self.estimate_error + self.process_variance

        if np.isnan(measurement):
            # If measurement is missing, increase uncertainty and use prediction
            self.estimate = prediction
            self.estimate_error = prediction_error
        else:
            # Update
            kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
            self.estimate = prediction + kalman_gain * (measurement - prediction)
            self.estimate_error = (1 - kalman_gain) * prediction_error

        self.last_estimate = self.estimate
        return self.estimate

def kalman_filter_data(data, process_variance, measurement_variance):
    # Find first non-NaN value for initialization
    initial_value = next((x for x in data if not np.isnan(x)), 0)
    
    kf = KalmanFilter(process_variance, measurement_variance, initial_value)
    filtered_data = np.zeros_like(data)
    for i, measurement in enumerate(data):
        filtered_data[i] = kf.update(measurement)
    return filtered_data

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
    
    for i, (column_name, sigma, edge_threshold, process_variance, measurement_variance) in enumerate(params):
        data = df[column_name].values
        
        # Apply Kalman filter
        kalman_filtered = kalman_filter_data(data, process_variance, measurement_variance)
        
        # Apply Gaussian filter
        gaussian_filtered = adaptive_gaussian_filter(kalman_filtered, sigma, edge_threshold)
        
        filtered_data[f'filtered_{column_name}'] = gaussian_filtered
        
        axs[i].plot(df['x'], data, label='Original', alpha=0.5)
        axs[i].plot(df['x'], kalman_filtered, label='Kalman Filtered', alpha=0.7)
        axs[i].plot(df['x'], gaussian_filtered, label='Kalman + Gaussian', linewidth=2)
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

# Parameters: (column_name, sigma, edge_threshold, process_variance, measurement_variance)
params = [
    ('lateral_distance', 3, 1, 0.1, 1),
    ('d', 3, 1, 0.1, 1),
    ('a_alks', 3, 1, 0.1, 1),
    ('lateral_movement', 3, 1, 0.1, 1),
    ('d_lateral_distance', 2, 1, 0.1, 1),
    ('ttc', 0, 0, 0.1, 1)
]

input_path = 'output.csv'
output_path = 'filtered_output.csv'
filtered_data = process_multiple_columns(input_path, params, output_path)

print(f"Filtered data has been saved to {output_path}")