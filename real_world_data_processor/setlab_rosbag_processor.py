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
from scipy.interpolate import UnivariateSpline, CubicSpline


@dataclass
class BagReaderConfig:
    storage_id: str = 'mcap'
    input_format: str = 'cdr'
    output_format: str = 'cdr'


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

        while reader.has_next():
            try:
                topic, data, t = reader.read_next()
                if topic in topics:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    self.messages.append((topic, msg, t))
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
                    # 'orientation_x': obj.kinematics.pose_with_covariance.pose.orientation.x,
                    # 'orientation_y': obj.kinematics.pose_with_covariance.pose.orientation.y,
                    # 'orientation_z': obj.kinematics.pose_with_covariance.pose.orientation.z,
                    # 'orientation_w': obj.kinematics.pose_with_covariance.pose.orientation.w,
                    # 'actor_speed': obj.kinematics.twist.twist.linear.x,
                }
                objects_data.append(obj_data)
        return objects_data


class DataProcessor:
    """Handles data processing and calculations."""

    @staticmethod
    def adaptive_gaussian_filter(data: np.ndarray, sigma: float, 
                               edge_threshold: float) -> np.ndarray:
        """Apply adaptive Gaussian filtering to the data."""
        if sigma == 0:
            return data
        filtered = gaussian_filter(data, sigma)
        gradient = np.abs(np.gradient(data))
        edge_mask = gradient > (edge_threshold * np.max(gradient))
        result = np.where(edge_mask, data, filtered)
        return result

    def process_multiple_columns(self, df: pd.DataFrame, 
                               sigma: float = 15, 
                               edge_threshold: float = 4) -> pd.DataFrame:
        """Process multiple columns with adaptive Gaussian filtering."""
        original_df = df.copy()
        n_cols = len(df.columns[1:])
        fig, axs = plt.subplots(n_cols, 1, figsize=(12, 6*n_cols), sharex=True)
        if n_cols == 1:
            axs = [axs]
        
        for i, column_name in enumerate(df.columns[1:]):
            data = df[column_name].values
            filtered = self.adaptive_gaussian_filter(data, sigma, edge_threshold)
            df[column_name] = filtered
            
            axs[i].plot(df['time'], data, label='Original', alpha=0.5)
            axs[i].plot(df['time'], filtered, label='Filtered', linewidth=2)
            axs[i].set_title(f'{column_name} (σ={sigma}, edge_threshold={edge_threshold})')
            axs[i].set_ylabel(column_name)
            axs[i].legend()
            axs[i].grid(True)
        
        axs[-1].set_xlabel('time')
        plt.tight_layout()
        plt.show()

        for col, filtered in df.items():
            original_df[col] = filtered

        return original_df

    @staticmethod
    def synchronize_timeseries(df: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame:
        """Synchronize multiple time series data."""
        df = df.copy()
        original_time = df[time_column].copy()
    
        df[time_column] = pd.to_datetime(df[time_column], unit='s')
        df.set_index(time_column, inplace=True)
        
        velocity_frequency = df['ego_speed'].dropna().index.to_series().diff().mean()
        acceleration_frequency = df['ego_acc_x'].dropna().index.to_series().diff().mean()
        ego_position_frequency = df['ego_x'].dropna().index.to_series().diff().mean()
        obj_position_frequency = df['obj_x'].dropna().index.to_series().diff().mean()
            
        fastest_frequency = min(velocity_frequency, ego_position_frequency, 
                              obj_position_frequency, acceleration_frequency)
        
        ms = fastest_frequency.total_seconds()*1000

        data_resampled = df.resample(f'{ms}ms').mean()
        data_interpolated = data_resampled.interpolate(method='spline', 
                                                     order=3).bfill()

        data_interpolated.reset_index(inplace=True)
        data_interpolated[time_column] = original_time
            
        return data_interpolated

    @staticmethod
    def calculate_obj_vehicle_velocity(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate object vehicle velocity."""
        df['obj_abs_x'] = df['ego_x'] + df['obj_x']
        df['obj_abs_y'] = df['ego_y'] + df['obj_y']
        
        df['time_diff'] = df['time'].diff()
        df['x_diff'] = df['obj_abs_x'].diff()
        df['y_diff'] = df['obj_abs_y'].diff()
        
        df['actor_speed'] = df['x_diff']  / df['time_diff']

        return df

    @staticmethod
    def calculate_observables(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate observable metrics from the data."""
        df['longituidinal_distance'] = df['obj_x']
        df['lateral_distance'] = df['obj_y']
        
        t = df['time'].diff()

        df = DataProcessor.calculate_obj_vehicle_velocity(df)

        # df['distance_from_midline'] = df['obj_y'] / 2
        df['ego_acceleration_1'] = (df['ego_speed'].diff()) / t
        df['ego_acceleration_2'] = np.sqrt(df['ego_acc_x']**2 + df['ego_acc_y']**2)
        df['ego_acceleration_1'] = df['ego_acceleration_1'].fillna(0)
        df['derivative_lateral_distance'] = (df['lateral_distance'].diff()) / t
        df['derivative_lateral_distance'] = df['derivative_lateral_distance'].fillna(0)
        df['time_to_colision'] = df['longituidinal_distance'] / (df['ego_speed'] - df['actor_speed'])
        df['time_to_colision'] = df['time_to_colision'].fillna(np.inf)

        df = df.drop(['ego_x', 'ego_y', 'obj_abs_y', 'obj_abs_x',
                    'time_diff', 'x_diff', 'y_diff', 'obj_x', 'obj_y'], axis=1)
        return df


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


class SetlabsRosbagProcessor:
    """Main class for processing ROS bags."""

    def __init__(self):
        self.reader = RosbagReader()
        self.extractor = DataExtractor()
        self.processor = DataProcessor()
        self.visualizer = Visualizer()

    def process_bag(self, bag_path: str, output_path: str) -> None:
        """Process the ROS bag and save results to CSV."""
        ego_topics = [
            '/sensing/gnss/pose_with_covariance',
            '/vehicle/twist',
            '/sensing/imu/imu_data'
        ]
        object_topic = '/perception/object_recognition/detection/apollo/objects'

        self.reader.read_messages(bag_path, ego_topics + [object_topic])
        ego_data_list = []
        obj_data_list = []

        for topic, msg, t in self.reader.messages:
            if topic in ego_topics:
                ego_data = self.extractor.extract_ego_data(msg, topic)
                if ego_data:
                    ego_data_list.append(ego_data)
            elif topic == object_topic:
                obj_data = self.extractor.extract_object_data(msg)
                obj_data_list.extend(obj_data)

        # Create DataFrames
        ego_df = pd.DataFrame(ego_data_list)
        obj_df = pd.DataFrame(obj_data_list)

        # Find the first index in ego_df where ego_speed is not 0 or NaN
        non_zero_index = ego_df.loc[ego_df['ego_speed'] > 0].index.min()

        # Filter ego_df: Keep only rows starting from the first non-zero ego_speed
        filtered_ego_df = ego_df.loc[non_zero_index:].reset_index(drop=True)
        filtered_ego_df = filtered_ego_df.sort_values(by='time', ascending=True).reset_index(drop=True)

        # Get the minimum time from the filtered ego data
        min_time = filtered_ego_df['time'].min()

        # Filter obj_df: Keep only rows with time >= min_time
        filtered_obj_df = obj_df.loc[obj_df['time'] >= min_time].reset_index(drop=True)
        filtered_obj_df = filtered_obj_df.sort_values(by='time', ascending=True).reset_index(drop=True)

        # Concatenate ego and obj dataframes
        merged_df = pd.concat([ego_df, obj_df], ignore_index=True)

        # Sort the merged DataFrame by the 'time' column
        merged_df = merged_df.sort_values(by='time', ascending=True).reset_index(drop=True)

        merged_df = self.processor.synchronize_timeseries(merged_df)
        merged_df = self.processor.process_multiple_columns(merged_df)
        merged_df = self.processor.calculate_observables(merged_df)
        self.visualizer.plot_all_columns_against_time(merged_df)
        # processed_df = self.processor.process_multiple_columns(merged_df)

        # delta_t = merged_df['time'].diff()

        # fig, axs = plt.subplots(1, 1, sharex=True)
        # axs.plot(delta_t, label='Original', alpha=0.5)
        # axs.set_ylabel('delta')
        # axs.legend()
        # axs.grid(True)
        # plt.tight_layout()
        # plt.show()

        ego_df.to_csv(output_path, index=False)
        # Save to an Excel file with multiple sheets
        print(f"CSV file created successfully: {output_path}")


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