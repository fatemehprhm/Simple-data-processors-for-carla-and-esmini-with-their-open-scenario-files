import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import StorageOptions, ConverterOptions
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

def read_messages(bag_path, topics):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr',
                                         output_serialization_format='cdr')

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    messages = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        try:
            if topic in topics:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                messages.append((topic, msg, t))
        except Exception as e:
            pass
    return messages

def extract_ego_data(msg):
    return {
        'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        'ego_x': msg.pose.pose.position.x,
        'ego_y': msg.pose.pose.position.y,
        'ego_twist_x': msg.twist.twist.linear.x,
        'ego_twist_y': msg.twist.twist.linear.y,
    }

def extract_object_data(msg):
    objects_data = []
    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    
    print(f"Message timestamp: {timestamp}")
    
    for obj in msg.objects:
        obj_data = {
            'timestamp': timestamp,
            'obj_x': obj.pose.position.x,
            'obj_y': obj.pose.position.y,
            'obj_twist_x': obj.twist.linear.x,
            'obj_twist_y': obj.twist.linear.y,
            'classification': obj.classification,
        }
        objects_data.append(obj_data)
    
    if not objects_data:
        print("Warning: No objects found in this message")
    
    return objects_data

def process_bag(bag_path, output_csv):
    ego_topic = '/carla/ego_vehicle/odometry'
    object_topic = '/carla/ego_vehicle/objects'

    messages = read_messages(bag_path, [ego_topic, object_topic])
    ego_data = []
    objects_data = []

    for topic, msg, t in messages:
        if topic == ego_topic:
            data = extract_ego_data(msg)
            if data:
                ego_data.append(data)
        if topic == object_topic:
            objects_data.extend(extract_object_data(msg))

    df_ego = pd.DataFrame(ego_data)
    df_obj = pd.DataFrame(objects_data)

    print("Ego data shape:", df_ego.shape)
    print("Object data shape:", df_obj.shape)

    if df_obj.empty:
        print("Error: No object data found. Cannot proceed with merging and calculations.")
        return

    # Merge ego data
    df_ego = df_ego.groupby('timestamp').first().reset_index()

    # Find the closest ego timestamp for each object timestamp
    def find_closest_timestamp(obj_time, ego_times):
        return ego_times.iloc[(ego_times - obj_time).abs().argsort()[0]]

    df_obj['ego_timestamp'] = df_obj['timestamp'].apply(
        lambda x: find_closest_timestamp(x, df_ego['timestamp']))

    # Merge object data with ego data
    merged_df = pd.merge_asof(df_obj.sort_values('timestamp'), 
                              df_ego.sort_values('timestamp'), 
                              left_on='ego_timestamp', 
                              right_on='timestamp', 
                              direction='nearest', 
                              suffixes=('_obj', '_ego'))

    
    merged_df['time_diff'] = merged_df['timestamp_obj'].diff()

    car_length = 3

    merged_df['ego_speed'] = np.sqrt(merged_df['ego_twist_x']**2 + merged_df['ego_twist_y']**2)

    merged_df['ego_acceleration'] = (merged_df['ego_x'] - 2 * merged_df['ego_x'].shift(1) + merged_df['ego_x'].shift(2)) / (merged_df['time_diff']**2)
    merged_df['ego_acceleration'] = merged_df['ego_acceleration'].fillna(0)

    merged_df['actor_speed'] = np.sqrt(merged_df['obj_twist_x']**2 + merged_df['obj_twist_y']**2)

    merged_df['actor_acceleration'] = (merged_df['obj_x'] - 2 * merged_df['obj_x'].shift(1) + merged_df['obj_x'].shift(2)) / (merged_df['time_diff']**2)
    merged_df['actor_acceleration'] = merged_df['actor_acceleration'].fillna(0)

    merged_df['longituidinal_distance'] = np.sqrt((merged_df['obj_x'] - merged_df['ego_x'])**2 + 
                                    (merged_df['obj_y'] - merged_df['ego_y'])**2) - car_length

    merged_df.rename(columns={'timestamp_obj': 'time'}, inplace=True)

    # Save to CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"CSV file created successfully: {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process ROS2 MCAP bag for cut-in scenario.")
    parser.add_argument("--bag", required=True, help="Input MCAP bag file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")

    args = parser.parse_args()

    process_bag(args.bag, args.output)