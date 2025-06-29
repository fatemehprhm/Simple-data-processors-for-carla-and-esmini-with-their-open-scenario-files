import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from derived_object_msgs.msg import ObjectArray
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class BagSubscriber(Node):
    def __init__(self):
        super().__init__('bag_subscriber')
        self.ego_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.ego_callback,
            10)
        self.object_sub = self.create_subscription(
            ObjectArray,
            '/carla/ego_vehicle/objects',
            self.object_callback,
            10)
        
        self.ego_data = []
        self.objects_data = []
        self.last_msg_time = time.time()
        self.timeout = 5.0 

    def ego_callback(self, msg):
        ego_info = self.extract_ego_data(msg)
        self.ego_data.append(ego_info)
        self.last_msg_time = time.time()

    def object_callback(self, msg):
        objects_info = self.extract_object_data(msg)
        self.objects_data.extend(objects_info)
        self.last_msg_time = time.time()

    def extract_ego_data(self, msg):
        return {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'ego_x': msg.pose.pose.position.x,
            'ego_y': msg.pose.pose.position.y,
            'ego_twist_x': msg.twist.twist.linear.x,
            'ego_twist_y': msg.twist.twist.linear.y,
        }

    def extract_object_data(self, msg):
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

def process_data(ego_data, objects_data, output_csv):
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

    t = merged_df['timestamp_obj'].diff()

    car_length = 3

    merged_df['ego_speed'] = np.sqrt(merged_df['ego_twist_x']**2 + merged_df['ego_twist_y']**2)

    merged_df['ego_acceleration'] = (merged_df['ego_x'] - 2 * merged_df['ego_x'].shift(1) + merged_df['ego_x'].shift(2)) / (t**2)
    merged_df['ego_acceleration'] = merged_df['ego_acceleration'].fillna(0)

    merged_df['actor_speed'] = np.sqrt(merged_df['obj_twist_x']**2 + merged_df['obj_twist_y']**2)

    merged_df['actor_acceleration'] = (merged_df['obj_x'] - 2 * merged_df['obj_x'].shift(1) + merged_df['obj_x'].shift(2)) / (t**2)
    merged_df['actor_acceleration'] = merged_df['actor_acceleration'].fillna(0)

    merged_df['longituidinal_distance'] = np.sqrt((merged_df['obj_x'] - merged_df['ego_x'])**2 + 
                                    (merged_df['obj_y'] - merged_df['ego_y'])**2) - car_length

    merged_df.rename(columns={'timestamp_obj': 'time'}, inplace=True)

    merged_df_new = merged_df.drop(['ego_timestamp', 'ego_x', 'ego_y', 'ego_twist_x', 'ego_twist_y', 'time_diff',
                                    'timestamp_obj', 'obj_x', 'obj_y', 'obj_twist_x', 'obj_twist_y', 'classification'], axis=1)

    # Save to CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"CSV file created successfully: {output_csv}")

def main(args=None):
    rclpy.init(args=args)
    bag_subscriber = BagSubscriber()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(bag_subscriber, timeout_sec=0.5)
            if time.time() - bag_subscriber.last_msg_time > bag_subscriber.timeout:
                print("No messages received for 5 seconds. Assuming bag playback is complete.")
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping...")
    finally:
        print("Processing collected data...")
        process_data(bag_subscriber.ego_data, bag_subscriber.objects_data, 'fv_output.csv')
        
        bag_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()