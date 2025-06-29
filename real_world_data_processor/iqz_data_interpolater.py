import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

data = pd.read_csv('iqz_data/fv_output.csv')

time = data['time'].values
columns_to_interpolate = ['ego_speed', 'ego_acceleration', 'actor_speed', 'actor_acceleration', 'longituidinal_distance']

target_time_rate = 0.1
new_time = np.arange(time[0], time[-1], target_time_rate)

interpolated_data = {}
for column in columns_to_interpolate:
    interpolation_function = interp1d(time, data[column].values, fill_value="extrapolate")
    interpolated_data[column] = interpolation_function(new_time)

interpolated_df = pd.DataFrame({
    'time': new_time,
    'ego_speed': interpolated_data['ego_speed'],
    'ego_acceleration': interpolated_data['ego_acceleration'],
    'actor_speed': interpolated_data['actor_speed'],
    'actor_acceleration': interpolated_data['actor_acceleration'],
    'longituidinal_distance': interpolated_data['longituidinal_distance']
})

def apply_kalman_filter(observations):
    kf = KalmanFilter(initial_state_mean=observations[0], n_dim_obs=1)
    kalman_smoothed, _ = kf.smooth(observations)
    return kalman_smoothed.flatten()

for column in columns_to_interpolate:
    interpolated_df[column] = apply_kalman_filter(interpolated_df[column])

plt.figure(figsize=(10, 6))
plt.plot(new_time, interpolated_data['ego_speed'], label="Interpolated", linestyle='--', color='gray')
plt.plot(new_time, interpolated_df['ego_speed'], label="Kalman Filtered", color='blue')
plt.xlabel('Time (seconds)')
plt.ylabel('Ego Speed')
plt.title('Ego Speed - Interpolated vs. Kalman Filtered')
plt.legend()
plt.grid(True)
plt.show()

interpolated_df.to_csv('iqz_data/fv_filtered_output.csv', index=False)
