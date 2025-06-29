## Object detection using YOLOv8
In order to run object detection node, build the ros package in the main directory and source it.
```
$ colcon build
source install/setup.zsh
```
In another terminal play the rosbag node which publishes image data.
```
$ ros2 bag play -s mcap cutin.mcap
```
Now open rviz2 and subscribe to ***inference result Image*** topic and run this program in the first terminal.
```
$ python3 src/object_detection/object_detection/yolov8_ros2_pt.py
```