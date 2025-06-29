import pandas as pd
from typing import Any, Dict, Optional, List

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


import os  # Import os for directory operations

class DataExtractor:
    """Handles extraction of data from ROS messages and calculates overall speed and acceleration."""

    def __init__(self):
        # Dictionary to store processed data by topic
        self.data_by_topic = {}

    @staticmethod
    def extract_ego_data(msg: Any, topic: str) -> Optional[Dict[str, float]]:
        """Extract ego vehicle data from GNSS, IMU, and twist messages."""
        if topic == '/sensing/imu/imu_data':
            return {
                'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'ego_acc_x': msg.linear_acceleration.x,
                'ego_acc_y': msg.linear_acceleration.y,
            }
        elif topic == '/vehicle/twist':
            return {
                'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'ego_speed_x': msg.twist.linear.x,
                'ego_speed_y': msg.twist.linear.y,
            }
        if topic == '/sensing/gnss/pose_with_covariance':
            return {
                'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'ego_x': msg.pose.pose.position.x,
                'ego_y': msg.pose.pose.position.y,
            }
        if topic == '/vehicle/gps/fix':
            return {
                'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'ego_latitude': msg.latitude,
                'ego_longitude': msg.longitude,
            }
        if topic == '/vehicle/imu/data_raw':
            return {
                'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'ego_acc_x': msg.linear_acceleration.x,
                'ego_acc_y': msg.linear_acceleration.y,
            }
        return None

    def process_message(self, msg: Any, topic: str):
        """Process a single message and append the extracted data."""
        extracted_data = self.extract_ego_data(msg, topic)
        if extracted_data:
            # Calculate overall speed or acceleration if applicable
            if 'ego_speed_x' in extracted_data and 'ego_speed_y' in extracted_data:
                extracted_data['ego_speed'] = np.sqrt(
                    extracted_data['ego_speed_x']**2 + extracted_data['ego_speed_y']**2
                )
            if 'ego_acc_x' in extracted_data and 'ego_acc_y' in extracted_data:
                extracted_data['ego_acc'] = np.sqrt(
                    extracted_data['ego_acc_x']**2 + extracted_data['ego_acc_y']**2
                )
            if 'ego_x' in extracted_data and 'ego_y' in extracted_data:
                extracted_data['ego_distance'] = np.sqrt(
                    extracted_data['ego_x']**2 + extracted_data['ego_y']**2
                )
            if 'ego_latitude' in extracted_data and 'ego_longitude' in extracted_data:
                extracted_data['ego_distance'] = np.sqrt(
                    extracted_data['ego_latitude']**2 + extracted_data['ego_longitude']**2
                )

            # Store data grouped by topic
            if topic not in self.data_by_topic:
                self.data_by_topic[topic] = []
            self.data_by_topic[topic].append(extracted_data)

    def save_to_csv_by_topic(self, output_dir: str):
        """Save the collected data for each topic to separate CSV files."""
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for topic, data in self.data_by_topic.items():
            # Generate a safe filename from the topic name
            filename = f"{output_dir}/{topic.replace('/', '_').strip('_')}.csv"
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"Data for topic '{topic}' saved to {filename}")



# Specify the topics you are interested in
ego_topics = [
    '/vehicle/gps/fix',
]

# Read messages from the bag
reader = RosbagReader()
reader.read_messages('setlabs_data/cutin.mcap', ego_topics)

# Process and save messages
data_extractor = DataExtractor()
for topic, msg, t in reader.messages:
    data_extractor.process_message(msg, topic)

# Save each topic's data to separate CSV files
data_extractor.save_to_csv_by_topic("output")