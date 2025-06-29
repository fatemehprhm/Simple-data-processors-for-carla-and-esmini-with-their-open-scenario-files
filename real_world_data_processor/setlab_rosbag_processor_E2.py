from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import StorageOptions, ConverterOptions
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


@dataclass
class BagReaderConfig:
    storage_id: str = 'mcap'
    input_format: str = 'cdr'
    output_format: str = 'cdr'

@dataclass
class State:
    time: float
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float
    acc_x: float
    acc_y: float

class ImuIntegrator:
    def __init__(self, position_confidence_threshold: float = 0.5):
        self.states: List[State] = []
        self.last_correction_time = 0
        self.position_confidence_threshold = position_confidence_threshold
        self.time_not_calculated = True
        
    def estimate_tilt(self, initial_measurements: List[Dict[str, float]], 
                     window_size: int = 1000) -> float:
        """Estimate IMU tilt from initial steady-state measurements"""
        steady_acc_y = [msg['ego_acc_y'] for msg in initial_measurements[:window_size]]
        median_acc_y = np.median(steady_acc_y)
        return np.arcsin(median_acc_y / 9.81)

    def correct_acceleration(self, acc_x: float, acc_y: float, tilt: float) -> Tuple[float, float]:
        """Remove gravity component and correct for tilt"""
        cos_tilt = np.cos(tilt)
        sin_tilt = np.sin(tilt)
        
        acc_x_corrected = acc_x * cos_tilt - acc_y * sin_tilt
        acc_y_corrected = acc_x * sin_tilt + acc_y * cos_tilt - 9.81 * sin_tilt
        
        return acc_x_corrected, acc_y_corrected

    def integrate_step(self, initial_gnss, imu_msg: Dict[str, float], tilt: float, mean_dt) -> None:
        """Integrate one IMU measurement to update position and velocity"""
        initial_position_x = initial_gnss['ego_x']
        initial_position_y = initial_gnss['ego_y']
        time = imu_msg['time']
        acc_x_raw = imu_msg['ego_acc_x']
        acc_y_raw = imu_msg['ego_acc_y']
        
        # Correct accelerations for tilt
        acc_x, acc_y = self.correct_acceleration(acc_x_raw, acc_y_raw, tilt)
        
        if not self.states:
            # Initialize first state
            self.states.append(State(
                time=initial_gnss['time'],
                position_x=initial_position_x,
                position_y=initial_position_y,
                velocity_x=0.0,
                velocity_y=0.0,
                acc_x=acc_x,
                acc_y=acc_y
            ))
            return
        
            
        prev_state = self.states[-1]
        dt = time - prev_state.time

        factor = 1.8

        if self.time_not_calculated:
            self.expected_dt = mean_dt * factor
            self.time_not_calculated = False
        
        # Check for missing data (gap larger than expected)
        if dt > self.expected_dt:
            # Calculate number of missing steps
            n_steps = int(dt * factor / self.expected_dt) - 1
            # Create intermediate states
            # print("timestamp", timestamp)
            # print("prev timestamp", prev_state.timestamp)
            for i in range(n_steps):
                step_dt = self.expected_dt / factor
                intermediate_time = prev_state.time + step_dt * (i + 1)
                # print("intermediate timestamp", intermediate_time)
                # Linearly interpolate acceleration
                alpha = (i + 1) / (n_steps + 1)
                intermediate_acc_x = prev_state.acc_x * (1 - alpha) + acc_x * alpha
                intermediate_acc_y = prev_state.acc_y * (1 - alpha) + acc_y * alpha
                
                # Calculate intermediate velocity
                intermediate_vel_x = prev_state.velocity_x + intermediate_acc_x * step_dt
                intermediate_vel_y = prev_state.velocity_y + intermediate_acc_y * step_dt
                
                # Calculate intermediate position
                intermediate_pos_x = prev_state.position_x + intermediate_vel_x * step_dt
                intermediate_pos_y = prev_state.position_y + intermediate_vel_y * step_dt
                
                # Add intermediate state
                if intermediate_time < time:
                    # print("accepted")
                    self.states.append(State(
                        time=intermediate_time,
                        position_x=intermediate_pos_x,
                        position_y=intermediate_pos_y,
                        velocity_x=intermediate_vel_x,
                        velocity_y=intermediate_vel_y,
                        acc_x=intermediate_acc_x,
                        acc_y=intermediate_acc_y
                    ))
                
                # Update prev_state for next iteration
                prev_state = self.states[-1]
    
        # Now integrate the current measurement
        velocity_x = prev_state.velocity_x + acc_x * (time - prev_state.time)
        velocity_y = prev_state.velocity_y + acc_y * (time - prev_state.time)
        
        position_x = prev_state.position_x + velocity_x * (time - prev_state.time)
        position_y = prev_state.position_y + velocity_y * (time - prev_state.time)
        
        self.states.append(State(
            time=time,
            position_x=position_x,
            position_y=position_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            acc_x=acc_x,
            acc_y=acc_y
        ))

    def get_gnss_confidence(self, covariance: List[float]) -> Tuple[float, float]:
        """Calculate confidence scores from GNSS covariance"""
        # Extract position variances from covariance matrix
        var_x = covariance[0]  # [0,0] element
        var_y = covariance[7]  # [1,1] element
        
        # Convert variance to standard deviation and normalize to confidence score
        # Lower variance = higher confidence, clipped to [0,1]
        conf_x = np.clip(1.0 / (1.0 + np.sqrt(var_x)), 0, 1)
        conf_y = np.clip(1.0 / (1.0 + np.sqrt(var_y)), 0, 1)
        
        return conf_x, conf_y

    def correct_with_gnss(self, gnss_msg: Dict[str, float], covariance: List[float]) -> None:
        """Correct position drift using GNSS measurement, weighted by confidence"""
        time = gnss_msg['time']
        true_pos_x = gnss_msg['ego_x']
        true_pos_y = gnss_msg['ego_y']
        
        # Get confidence scores from covariance
        conf_x, conf_y = self.get_gnss_confidence(covariance)
        
        # Only apply corrections if confidence is above threshold
        if conf_x < self.position_confidence_threshold or conf_y < self.position_confidence_threshold:
            return
        
        # Find states since last correction
        correction_indices = []
        for i, state in enumerate(self.states):
            if self.last_correction_time < state.time <= time:
                correction_indices.append(i)
                
        if not correction_indices:
            return
            
        # Calculate weighted position error
        last_state = self.states[correction_indices[-1]]
        pos_error_x = (true_pos_x - last_state.position_x) * conf_x
        pos_error_y = (true_pos_y - last_state.position_y) * conf_y
        
        # Apply weighted corrections
        num_states = len(correction_indices)
        for idx, i in enumerate(correction_indices):
            fraction = idx / num_states
            correction_x = pos_error_x * fraction
            correction_y = pos_error_y * fraction
            
            self.states[i].position_x += correction_x
            self.states[i].position_y += correction_y
            
        self.last_correction_time = time


class RosbagReader:
    """Handles reading and deserializing messages from ROS2 bags."""
    
    def __init__(self, config: BagReaderConfig = BagReaderConfig()):
        self.config = config
        self.messages = []

    def read_messages(self, bag_path: str, topics: List[str]) -> None:
        """Read messages from specified topics in the bag file."""
        storage_options = StorageOptions(
            uri=bag_path,
            storage_id=self.config.storage_id
        )
        converter_options = ConverterOptions(
            input_serialization_format=self.config.input_format,
            output_serialization_format=self.config.output_format
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_types[i].name: topic_types[i].type 
                   for i in range(len(topic_types))}
        
        counter = 0
        while reader.has_next():
            try:
                topic, data, t = reader.read_next()
                if topic in topics:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    self.messages.append((topic, msg, t))
                    counter += 1
            except Exception:
                pass


class DataExtractor:
    """Handles extraction of data from ROS messages."""

    @staticmethod
    def extract_ego_data(msg: Any, topic: str) -> Optional[Dict[str, float]]:
        """Extract ego vehicle data from GNSS and twist messages."""
        if topic == '/sensing/gnss/pose_with_covariance':
            return {
                'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'ego_x': msg.pose.pose.position.x,
                'ego_y': msg.pose.pose.position.y,
                'covariance': msg.pose.covariance,
                # 'ego_z': msg.pose.pose.position.z,
            }
        elif topic == '/sensing/imu/imu_data':
            return {
                'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'ego_acc_x': msg.linear_acceleration.x,
                'ego_acc_y': msg.linear_acceleration.y,
            }
        elif topic == '/vehicle/twist':
            return {
                'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'ego_speed': msg.twist.linear.x,
                # 'ego_linear_velocity_y': msg.twist.linear.y,
                # 'ego_angular_velocity_z': msg.twist.angular.z,
            }
        return None

    @staticmethod
    def extract_object_data(msg: Any) -> List[Dict[str, float]]:
        """Extract data for detected objects."""
        objects_data = []
        for obj in msg.objects:
            if obj.classification and obj.classification[0].classification == 1:
                obj_data = {
                    'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    'obj_x': obj.kinematics.pose_with_covariance.pose.position.x,
                    'obj_y': obj.kinematics.pose_with_covariance.pose.position.y,
                }
                objects_data.append(obj_data)
        return objects_data

def adaptive_gaussian_filter(data: np.ndarray, sigma: float = 15, 
                               edge_threshold: float = 4) -> np.ndarray:
    """Apply adaptive Gaussian filtering to the data."""
    if sigma == 0:
        return data
    filtered = gaussian_filter(data, sigma)
    gradient = np.abs(np.gradient(data))
    edge_mask = gradient > (edge_threshold * np.max(gradient))
    result = np.where(edge_mask, data, filtered)
    return result

class Visualizer:
    """Handles data visualization."""

    @staticmethod
    def plot_all_columns_against_time(df: pd.DataFrame) -> None:
        """Plot all columns against time in separate subplots."""
        time_column = 'time'
        columns_to_plot = df.columns.drop(time_column)
        n_cols = len(columns_to_plot)

        fig, axs = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols), sharex=True)
        
        if n_cols == 1:
            axs = [axs]

        for i, col in enumerate(columns_to_plot):
            axs[i].plot(df[time_column], df[col], label=col)
            axs[i].set_ylabel(col)
            axs[i].legend()
            axs[i].grid(True)

        axs[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.show()

class ObjectCalculater:
    @staticmethod
    def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method."""
        mean = np.mean(data)
        std = np.std(data)
        z_scores = (data - mean) / std
        return np.abs(z_scores) > threshold

    @staticmethod
    def interpolate_missing_data(data: np.ndarray, method: str = 'cubic') -> np.ndarray:
        """Interpolate missing data using specified method."""
        if method == 'linear':
            return pd.Series(data).interpolate(method='linear').values
        elif method == 'cubic':
            return pd.Series(data).interpolate(method='cubic').values
        else:
            raise ValueError("Unsupported interpolation method")

    def process_object_position(self, df: pd.DataFrame, column: str = 'obj_x') -> pd.DataFrame:
        """Process object position data to remove outliers and interpolate missing values."""
        data = df[column].values
        
        # Detect outliers using Z-score or IQR
        outliers = self.detect_outliers_zscore(data)
        
        # Replace outliers with NaN
        data[outliers] = np.nan
        
        # Interpolate missing data
        data = self.interpolate_missing_data(data, method='cubic')
        
        # Update the DataFrame
        df[column] = data
        
        return df
    
    def calculate_obj_vehicle_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate object vehicle velocity."""
        df['obj_abs_x'] = df['position_x'] + df['obj_x']
        df['obj_abs_y'] = df['position_y'] + df['obj_y']
        
        df['time_diff'] = df['time'].diff()
        df['x_diff'] = df['obj_abs_x'].diff()
        df['y_diff'] = df['obj_abs_y'].diff()
        
        df['actor_speed'] = df['x_diff']  / df['time_diff']

        return df

    def calculate_observables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate observable metrics from the data."""
        line_width = 3.7
        car_width = 1.8
        df['longituidinal_distance'] = df['obj_x']
        df['lateral_distance'] = df['obj_y']
        
        t = df['time'].diff()

        df = self.calculate_obj_vehicle_velocity(df)

        df['distance_from_midline'] = df['obj_y'] - (line_width/2 - car_width/2)
        df['ego_acceleration'] = df['acc_x']
        df['derivative_lateral_distance'] = (df['lateral_distance'].diff()) / t
        df['derivative_lateral_distance'] = df['derivative_lateral_distance'].fillna(0)
        df['time_to_colision'] = df['longituidinal_distance'] / (df['velocity_x'] - df['actor_speed'])
        df['time_to_colision'] = df['time_to_colision'].fillna(np.inf)

        df = df.drop(['position_x', 'position_y', 'obj_abs_y', 'obj_abs_x',
                    'time_diff', 'x_diff', 'y_diff'], axis=1)
        return df

# Update `SetlabsRosbagProcessor`
class SetlabsRosbagProcessor:
    """Main class for processing ROS bags."""

    def __init__(self):
        self.reader = RosbagReader()
        self.extractor = DataExtractor()
        self.imu_integrator = ImuIntegrator()
        self.visualizer = Visualizer()
        self.object_calculater = ObjectCalculater()

    def process_bag(self, bag_path: str, output_path: str) -> None:
        """Process the ROS bag and save results to CSV."""
        ego_topics = [
            '/sensing/gnss/pose_with_covariance',
            '/vehicle/twist',
            '/sensing/imu/imu_data'
        ]
        object_topic = '/perception/object_recognition/detection/apollo/objects'

        self.reader.read_messages(bag_path, ego_topics + [object_topic])

        # Process messages with IMU integrator
        imu_messages = []
        gnss_messages = []
        object_messages = []

        for topic, msg, t in self.reader.messages:
            if topic == '/sensing/imu/imu_data':
                imu_msg = self.extractor.extract_ego_data(msg, topic)
                imu_messages.append(imu_msg)
            elif topic == '/sensing/gnss/pose_with_covariance':
                gnss_msg = self.extractor.extract_ego_data(msg, topic)
                gnss_messages.append(gnss_msg)
            elif topic == '/perception/object_recognition/detection/apollo/objects':
                object_msg = self.extractor.extract_object_data(msg)
                object_messages.extend(object_msg)

        # obj_df = pd.DataFrame(object_messages)
        # obj_df_test = pd.DataFrame(object_messages)

        # obj_df = self.object_calculater.process_object_position(obj_df, column='obj_x')
        # obj_df = self.object_calculater.process_object_position(obj_df, column='obj_y')

        # # Estimate initial tilt
        # tilt = self.imu_integrator.estimate_tilt(imu_messages[:100])

        # timestamps = [msg['time'] for msg in imu_messages[:20]]
        # time_differences = np.diff(timestamps)  # Calculate consecutive differences
        # mean_dt = np.mean(time_differences)
        
        # # Process messages in time order
        # gnss_idx = 0
        # for imu_msg in imu_messages:
        #     # Integrate IMU data
        #     self.imu_integrator.integrate_step(gnss_messages[0], imu_msg, tilt, mean_dt)
            
        #     # Apply GNSS correction when available
        #     while (gnss_idx < len(gnss_messages) and 
        #         gnss_messages[gnss_idx]['time'] <= imu_msg['time']):
        #         gnss_msg = gnss_messages[gnss_idx]
        #         self.imu_integrator.correct_with_gnss(gnss_msg, gnss_msg['covariance'])
        #         gnss_idx += 1
        
        # states = self.imu_integrator.states

        # Save results
        df = pd.DataFrame(imu_messages)
        # df['position_x'] = adaptive_gaussian_filter(df['position_x'])
        # df['position_y'] = adaptive_gaussian_filter(df['position_y'])
        
        # ego_timestamps = df['time'].values
        # ego_positions_x = df['position_x'].values
        # ego_positions_y = df['position_y'].values

        # # Extract cut-in vehicle data
        # obj_timestamps = obj_df['time'].values
        # obj_positions_x = obj_df['obj_x'].values
        # obj_positions_y = obj_df['obj_y'].values

        # # Interpolate cut-in vehicle positions to ego timestamps
        # interp_obj_x = interp1d(obj_timestamps, obj_positions_x, kind='linear', fill_value='extrapolate')
        # interp_obj_y = interp1d(obj_timestamps, obj_positions_y, kind='linear', fill_value='extrapolate')

        # df['obj_x'] = interp_obj_x(ego_timestamps)
        # df['obj_y'] = interp_obj_y(ego_timestamps)

        # df = self.object_calculater.calculate_observables(df)

        # self.visualizer.plot_all_columns_against_time(df)
        df.to_csv(output_path, index=False)


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(description="Process ROS2 MCAP bag for cut-in scenario.")
    parser.add_argument("--bag", required=True, help="Input MCAP bag file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")

    args = parser.parse_args()

    setlabs_rosbag_processor = SetlabsRosbagProcessor()
    setlabs_rosbag_processor.process_bag(args.bag, args.output)


if __name__ == "__main__":
    main()