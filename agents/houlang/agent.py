import math
import numpy as np
from sturnus.geo import *

class Agent():
    def __init__(self, side) -> None:
        self.side = side
        self.rng = np.random.default_rng()
        self.assign = {}
        self.missile_cds = {}
        self.missile_time = {}

        self.num_step = 0
        self.history = {}
        self.previous_controls = {}

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

    def calculate_direction(self, plane_info, target_info):
        dx = target_info.x - plane_info.x
        dy = target_info.y - plane_info.y
        dz = target_info.z - plane_info.z
        horizontal_distance = math.sqrt(dx ** 2 + dy ** 2)
        azimuth = math.atan2(dy, dx)
        elevation = math.atan2(dz, horizontal_distance)
        return azimuth, elevation

    def control_requ(self, obs):
        heat_zone_center = (0, 0)
        heat_zone_radius = 15000
        z_min, z_max = -2000, 0  # 调整至更接近高空
        raw_cmd_dict = {}

        for key, value in obs.my_planes.items():
            my_plane_info = value
            target_info = type('target', (object,), {'x': heat_zone_center[0], 'y': heat_zone_center[1], 'z': z_max})

            azimuth, elevation = self.calculate_direction(my_plane_info, target_info)
            distance_to_target = self.calculate_distance(my_plane_info, target_info)

            # 确定舵机和节流阀的控制
            if distance_to_target > heat_zone_radius:
                # 飞向热区阶段
                aileron = 1.0 if azimuth < 0 else -1.0  # 根据方向调整
                elevator = 1.0 if elevation < 0 else -1.0
                throttle = 1.0  # 最大节流
                rudder = 0.0  # 始终为0
            else:
                # 盘旋阶段
                aileron = -0.5 if azimuth < 0 else 0.5  # 小幅调整以维持盘旋
                elevator = -0.5 if elevation < 0 else 0.5
                throttle = 0.5  # 减少速度以维持盘旋
                rudder = 0.0  # 始终为0

            this_plane_control = {'control': [aileron, elevator, rudder, throttle]}
            raw_cmd_dict[key] = this_plane_control

            # 打印调试信息
            if self.num_step % 10 == 0 or self.num_step == 1:
                print(f"Step: {self.num_step}")
                print(f"Plane ID: {key}, My plane coordinates: (x: {my_plane_info.x}, y: {my_plane_info.y}, z: {my_plane_info.z}, yaw: {my_plane_info.yaw}, v_north: {my_plane_info.v_north}, v_east: {my_plane_info.v_east}, v_down: {my_plane_info.v_down})")
                print(f"Target coordinates: (x: {target_info.x}, y: {target_info.y}, z: {target_info.z})")
                print(f"Azimuth: {azimuth}, Elevation: {elevation}")
                print(f"Control: aileron={aileron}, elevator={elevator}, rudder={rudder}, throttle={throttle}")

        return raw_cmd_dict
