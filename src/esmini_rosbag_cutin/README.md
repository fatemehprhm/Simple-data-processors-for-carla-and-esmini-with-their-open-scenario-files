# Rosbag
In order to have a semi-online monitoring we have to generate a rosbag file from csv file generated before. The final rosbag contains the data of 5 observables including: time, ego_speed, ego_accel, lead_accel, distance in every 0.04 seconds.

## Generating a Rosbag file
In order to generate a rosbag file for an intended scenario, first you have to build the packages. Run the following commands in main directory. Replace .zsh with .bash if you use bash command-line shell.
```
$ source /opt/ros/foxy/setup.zsh
colcon build
source install/setup.zsh
```
Now you should start publishing the ros msgs by running the csv2rosbag python file. But before publishing data, in order to record data in a rosbag file, run the following command. Topic name is "csv_data".
```
$ ros2 bag record -o path/to/output_folder /csv_data 
```
Now in a new terminal after sourcing the setup files as mentioned earlier, run the following command. It takes the csv file address as an argument. Asample csv file  is provided.
```
$ python3 src/csv2rosbag/csv2rosbag/csv2rosbag.py --csv path/to/csv_file
```
This will publish data every 0.04 second. Now, data is being recorded in db3 format. You can enter Ctrl+C to stop recording after all the data has been published.

## Subscribing to Rosbag
In order to subscribe to rosbag file, after sourcing required setups run the following command.
```
$ ros2 bag play path/to/db3_file 
```
In another terminal run the following.
```
$ ros2 topic echo /csv_data
```
This will show the published data by rosbag.