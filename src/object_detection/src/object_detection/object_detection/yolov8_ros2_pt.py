#!/usr/bin/env python3

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


bridge = CvBridge()

class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        
        self.model = YOLO('./yolov8m.pt')
        self.rgb_subscriber = self.create_subscription(Image,'/sensing/camera/front/resize',self.camera_callback,10)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.rgb_image = None

    def camera_callback(self, msg):
        self.rgb_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.rgb_image is None:
            return
        results = self.model(self.rgb_image, verbose=False)
        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame)  
        self.img_pub.publish(img_msg)


if __name__ == '__main__':
    rclpy.init(args=None)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()
