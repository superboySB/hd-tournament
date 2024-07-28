import math
import numpy as np
from sturnus.geo import *

def normalize_angle(angle):
    """ 将角度归一化到 [-π, π] 区间。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

class Agent():
    def __init__(self, side) -> None:
        self.num_step = 0

    def step(self, obs):
        self.num_step += 1

        # 控制逻辑
        raw_cmd_dict = self.control_requ(obs)
        
        return raw_cmd_dict

    def calculate_distance(self, plane_info, target_info):
        return math.sqrt(
            (target_info.x - plane_info.x) ** 2 +
            (target_info.y - plane_info.y) ** 2 +
            (target_info.z - plane_info.z) ** 2
        )

    def calculate_direction(self, plane_info, target_info, debug=True):
        dx = target_info.x - plane_info.x
        dy = target_info.y - plane_info.y
        dz = target_info.z - plane_info.z
        horizontal_distance = math.sqrt(dx ** 2 + dy ** 2)
        azimuth = math.atan2(dy, dx)
        elevation = math.atan2(dz, horizontal_distance)

        if debug:
            print(f"\ndx: {dx}, dy: {dy}, dz: {dz}")
            print(f"horizontal_distance: {horizontal_distance}")
            print(f"Calculated azimuth (before normalization): {azimuth}, Current yaw: {plane_info.yaw}")

        # 规范化所有角度
        azimuth = normalize_angle(azimuth)
        current_yaw = normalize_angle(plane_info.yaw)

        # 计算角度差，并且再次规范化，以确保是最短路径的差异
        azimuth_diff = normalize_angle(azimuth - current_yaw)

        if debug:
            print(f"Normalized azimuth: {azimuth}, Normalized current yaw: {current_yaw}")
            print(f"Azimuth difference: {azimuth_diff}")

        return azimuth_diff, elevation

    def control_requ(self, obs):
        heat_zone_center = (0, 0)
        heat_zone_radius = 15000
        z_target = 0
        raw_cmd_dict = {}

        for key, value in obs.my_planes.items():
            my_plane_info = value
            target_info = type('target', (object,), {'x': heat_zone_center[0], 'y': heat_zone_center[1], 'z': z_target})

            azimuth, elevation = self.calculate_direction(my_plane_info, target_info)
            distance_to_target = self.calculate_distance(my_plane_info, target_info)

            altitude_error = z_target - my_plane_info.z
            elevation_correction = np.clip(altitude_error / 100, -1, 1)

            aileron = np.clip(azimuth / 30, -1, 1)
            elevator = np.clip(elevation_correction, -0.7, 0.7)

            if distance_to_target > heat_zone_radius:
                flight_phase = "Approach"
                rudder = 0.0
            else:
                flight_phase = "Circling"
                rudder = np.clip(azimuth / 50, -1, 1)

            throttle = 0.7
            this_plane_control = {'control': [aileron, elevator, rudder, throttle]}
            raw_cmd_dict[key] = this_plane_control

            # if self.num_step % 20 == 0:
            #     distance_to_center = self.calculate_distance(my_plane_info, type('target', (object,), {'x': 0, 'y': 0, 'z': 0}))
            #     print(f"\nStep: {self.num_step}, Flight Phase: {flight_phase}, Distance to Heat Zone Center: {distance_to_center:.2f}")
            #     print(f"Plane ID: {key}, My plane coordinates: (x: {my_plane_info.x}, y: {my_plane_info.y}, z: {my_plane_info.z}, \
            #         roll: {my_plane_info.roll}, pitch: {my_plane_info.pitch}, yaw: {my_plane_info.yaw}, \
            #             v_north: {my_plane_info.v_north}, v_east: {my_plane_info.v_east}, v_down: {my_plane_info.v_down})")
            #     print(f"Target coordinates: (x: {target_info.x}, y: {target_info.y}, z: {target_info.z})")
            #     print(f"Azimuth: {azimuth:.2f}, Elevation: {elevation:.2f}")
            #     print(f"Control: aileron={aileron}, elevator={elevator}, rudder={rudder}, throttle={throttle}")

        return raw_cmd_dict

