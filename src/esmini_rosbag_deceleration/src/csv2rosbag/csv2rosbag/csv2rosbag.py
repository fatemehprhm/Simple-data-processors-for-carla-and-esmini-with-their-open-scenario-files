#!/usr/bin/env python3

import csv
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import argparse
import time
from builtin_interfaces.msg import Time as TimeMsg


class CsvToRosbagNode(Node):
    def __init__(self):
        super().__init__('csv_to_rosbag_node')
        
        # Actor publishers
        self.actor_position_pub = self.create_publisher(Odometry, 'actor/position', 10)
        self.actor_accel_pub = self.create_publisher(Imu, 'actor/acceleration', 10)
        
        # Ego publishers
        self.ego_position_pub = self.create_publisher(Odometry, 'ego/position', 10)
        self.ego_accel_pub = self.create_publisher(Imu, 'ego/acceleration', 10)
    
    def create_header(self, timestamp_sec):
        """Create a ROS Header with the given timestamp"""
        header = Header()
        
        # Create Time message
        time_msg = TimeMsg()
        time_msg.sec = int(timestamp_sec)
        time_msg.nanosec = int((timestamp_sec - int(timestamp_sec)) * 1e9)
        
        header.stamp = time_msg
        header.frame_id = "map"
        return header

    def csv_to_rosbag(self, csv_filename):
        with open(csv_filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            self.interval = 0.04

            for row in csvreader:
                timestamp = float(row[0])
                ros_header = self.create_header(timestamp)

                actor_pos = Odometry()
                actor_pos.header = ros_header
                actor_pos.pose.pose.position.x = float(row[5])
                actor_pos.pose.pose.position.y = float(row[6])
                actor_pos.pose.pose.position.z = 0.0
                # Initialize covariance with zeros
                actor_pos.pose.covariance = [0.0] * 36
                self.actor_position_pub.publish(actor_pos)

                actor_accel = Imu()
                actor_accel.header = ros_header
                actor_accel.linear_acceleration.x = float(row[8])
                actor_accel.linear_acceleration.y = 0.0
                actor_accel.linear_acceleration.z = 0.0
                self.actor_accel_pub.publish(actor_accel)

                ego_pos = Odometry()
                ego_pos.header = ros_header
                ego_pos.pose.pose.position.x = float(row[1])
                ego_pos.pose.pose.position.y = float(row[2])
                ego_pos.pose.pose.position.z = 0.0
                # Initialize covariance with zeros
                ego_pos.pose.covariance = [0.0] * 36
                self.ego_position_pub.publish(ego_pos)

                ego_accel = Imu()
                ego_accel.header = ros_header
                ego_accel.linear_acceleration.x = float(row[4])
                ego_accel.linear_acceleration.y = 0.0
                ego_accel.linear_acceleration.z = 0.0
                self.ego_accel_pub.publish(ego_accel)

                self.get_logger().info(f"Published data at time: {timestamp}")
                time.sleep(self.interval)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Publish CSV data as ROS 2 messages.')
    parser.add_argument('--csv', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    csv_to_rosbag_node = CsvToRosbagNode()
    csv_to_rosbag_node.csv_to_rosbag(args.csv)

    rclpy.spin(csv_to_rosbag_node)
    csv_to_rosbag_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()