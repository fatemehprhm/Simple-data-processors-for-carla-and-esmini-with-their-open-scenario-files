'''
The data will be extracted from Carla World 
and saved to a csv file using this code.
'''

import carla
from carla import ColorConverter as cc

import pandas as pd
import numpy as np
import argparse
from argparse import RawTextHelpFormatter

import weakref
import time
import math
import cv2


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        bp.set_attribute('noise_alt_stddev', '0.0')
        bp.set_attribute('noise_alt_bias', '0.0')
        bp.set_attribute('noise_lat_stddev', '0.0')
        bp.set_attribute('noise_lat_bias', '0.0')
        bp.set_attribute('noise_lon_stddev', '0.0')
        bp.set_attribute('noise_lon_bias', '0.0')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        bp.set_attribute('noise_accel_stddev_x', '0.0')
        bp.set_attribute('noise_accel_stddev_y', '0.0')
        bp.set_attribute('noise_accel_stddev_z', '0.0')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.recording = False
        self.IM_WIDTH = 640
        self.IM_HEIGHT = 480
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = 'sensor.camera.rgb'
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        self.bp = bp_library.find(self.sensors)
        self.bp.set_attribute('image_size_x', f"{self.IM_WIDTH}")
        self.bp.set_attribute('image_size_y', f"{self.IM_HEIGHT}")
        self.bp.set_attribute('gamma', '2.2')
        self.bp.set_attribute('fov', '110')


    def set_sensor(self):
        self.sensor = self._parent.get_world().spawn_actor(
            self.bp,
            self._camera_transforms[self.transform_index][0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[self.transform_index][1])
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)
        return i3/255.0

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():

    description = ("Data Extractor")

    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--filename', help='Provide a filename for data storage')

    arguments = parser.parse_args()

    if arguments.filename:
        storagefile = arguments.filename + '.csv'
    else:
        storagefile = 'data.csv'
    
    player = None
    otherplayer = None
    gnss_sensor = None
    imu_sensor = None
    imu_sensor2 = None
    _info_text = []
    data_list = []

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(40.0)
    world = client.get_world()
    map = world.get_map()

    # Get the ego and leading vehicle
    while (player is None) or (otherplayer is None):

        if player is None:
            print("Waiting for the ego vehicle...")
        time.sleep(1)
        if otherplayer is None:
            print("Waiting for the leading vehicle...")
        time.sleep(1)
        
        possible_vehicles = world.get_actors().filter('vehicle.*')

        for vehicle in possible_vehicles:

            if vehicle.attributes['role_name'] == 'hero' and (player is None):
                print("Ego vehicle found")
                player = vehicle
            
            if vehicle.attributes['role_name'] != 'hero' and (otherplayer is None):
                print("leading vehicle found")
                otherplayer = vehicle

    # Initializing sensors attached to ego vehicle
    gnss_sensor = GnssSensor(player)
    imu_sensor = IMUSensor(player)
    camera_manager = CameraManager(player)
    camera_manager.set_sensor()
    imu_sensor2 = IMUSensor(otherplayer)

    print('Map: % 10s' % map.name.split('/')[-1])

    ego_x0 = player.get_transform()

    # Main loop
    flag = True

    while flag:

        vehicles = world.get_actors().filter('vehicle.*')
        player_name = otherplayer.type_id
        
        world_snapshot = world.get_snapshot()
        t_time = world_snapshot.timestamp.elapsed_seconds

        t = player.get_transform()
        v = player.get_velocity()

        t1 = otherplayer.get_transform()
        v1 = otherplayer.get_velocity()

        ego_name = get_actor_display_name(player, truncate=20)
        leading_name = get_actor_display_name(otherplayer, truncate=20)

        ego_speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        leading_speed = 3.6 * math.sqrt(v1.x**2 + v1.y**2 + v1.z**2)
        
        _info_text = [
            '',
            'Vehicle: % 20s' %  ego_name,
            'Speed:   % 15.0f km/h' % ego_speed,
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (imu_sensor.accelerometer),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (gnss_sensor.lat, gnss_sensor.lon)),
            '',
            'LeadingVehicle: % 20s' % leading_name,
            'Speed:   % 15.0f km/h' % leading_speed,
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t1.location.x, t1.location.y)),
            '',
            ]

        data = {
            'time': t_time,
            'ego_speed': round(ego_speed, 2), 
            'ego_accel': tuple(round(x, 2) for x in imu_sensor.accelerometer), 
            'ego_location': (round(t.location.x, 2), round(t.location.y, 2)), 
            'ego_gnss': (round(gnss_sensor.lat, 6), round(gnss_sensor.lon, 6)),  
            'leading_speed': round(leading_speed, 2),
            'leading_accel': tuple(round(x, 2) for x in imu_sensor2.accelerometer), 
            'leading_location': (round(t1.location.x, 2), round(t1.location.y, 2))}

        if len(vehicles) > 1:
            _info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    data['distance'] = None
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                _info_text.append('% 4dm %s' % (d, vehicle_type))
                data['distance'] = round(d, 2)
        
        distance_traveled = math.sqrt((t.location.x - ego_x0.location.x)**2 + (t.location.y - ego_x0.location.y)**2 + (t.location.z - ego_x0.location.z)**2)
        data['distance traveled by Ego'] = round(distance_traveled, 2)

        data_list.append(data)

        for line in _info_text:
            print(line)

        print("-" * 40)

        time.sleep(0.04)

        if len(world.get_actors().filter(player_name)) < 1:
            gnss_sensor.sensor.destroy()
            imu_sensor.sensor.destroy()
            camera_manager.sensor.destroy()
            flag = False

    df = pd.DataFrame.from_dict(data_list)
    df.to_csv(storagefile)

if __name__ == '__main__':

    main()