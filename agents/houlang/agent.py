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
        self.pid_controller = PIDController(kp=0.01, ki=0.005, kd=0.001)

    def step(self, obs):
        self.num_step += 1

        # 进行热区控制
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

    def calculate_control(self, plane_info, target_info, azimuth, elevation, key):
        target_heading = math.degrees(azimuth)
        target_pitch = math.degrees(elevation)

        current_heading = plane_info.yaw
        current_pitch = plane_info.pitch

        heading_diff = (target_heading - current_heading + 180) % 360 - 180
        pitch_diff = target_pitch - current_pitch

        aileron = self.pid_controller.update(heading_diff, key, 'aileron')
        elevator = self.pid_controller.update(pitch_diff, key, 'elevator')
        rudder = aileron
        throttle = 0.7

        # 动态调整控制量
        aileron = max(min(aileron, 0.3), -0.3)
        elevator = max(min(elevator, 0.5), -0.5)
        rudder = max(min(rudder, 0.3), -0.3)
        throttle = max(min(throttle, 1), 0)

        prev_aileron, prev_elevator, prev_rudder, prev_throttle = self.previous_controls.get(key, [0, 0, 0, 0.7])
        smooth_factor = 0.2
        aileron = prev_aileron * (1 - smooth_factor) + aileron * smooth_factor
        elevator = prev_elevator * (1 - smooth_factor) + elevator * smooth_factor
        rudder = prev_rudder * (1 - smooth_factor) + rudder * smooth_factor
        throttle = prev_throttle * (1 - smooth_factor) + throttle * smooth_factor

        self.previous_controls[key] = [aileron, elevator, rudder, throttle]

        return [aileron, elevator, rudder, throttle]

    def control_requ(self, obs):
        heat_zone_center = (0, 0)
        heat_zone_radius = 15000
        z_min, z_max = -2000, 4000
        raw_cmd_dict = {}

        for key, value in obs.my_planes.items():
            my_plane_info = value

            # 计算到热区中心的方向
            target_info = type('target', (object,), {'x': heat_zone_center[0], 'y': heat_zone_center[1], 'z': 0})()
            azimuth, elevation = self.calculate_direction(my_plane_info, target_info)

            # 动态调整目标高度，确保平滑过渡
            distance_to_target = self.calculate_distance(my_plane_info, target_info)
            if distance_to_target < heat_zone_radius:
                target_info.z = z_max - (z_max - z_min) * (distance_to_target / heat_zone_radius)
            else:
                target_info.z = (z_min + z_max) / 2

            if my_plane_info.z < z_min:
                elevation = math.atan2(z_min - my_plane_info.z, heat_zone_radius)
            elif my_plane_info.z > z_max:
                elevation = math.atan2(z_max - my_plane_info.z, heat_zone_radius)

            control = self.calculate_control(my_plane_info, target_info, azimuth, elevation, key)
            
            this_plane_control = {'control': control}
            raw_cmd_dict[key] = this_plane_control

            # 打印调试信息
            if self.num_step % 10 == 0:
                print(f"Step: {self.num_step}")
                print(f"Plane ID: {key}, My plane coordinates: (x: {my_plane_info.x}, y: {my_plane_info.y}, z: {my_plane_info.z}, yaw: {my_plane_info.yaw}, v_north: {my_plane_info.v_north}, v_east: {my_plane_info.v_east}, v_down: {my_plane_info.v_down})")
                print(f"Control: aileron={control[0]}, elevator={control[1]}, rudder={control[2]}, throttle={control[3]}")

        return raw_cmd_dict

class PIDController:
    def __init__(self, kp=0.01, ki=0.005, kd=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = {}
        self.previous_error = {}

    def update(self, error, key, control_type):
        if key not in self.integral:
            self.integral[key] = {}
        if key not in self.previous_error:
            self.previous_error[key] = {}

        self.integral[key][control_type] = self.integral[key].get(control_type, 0) + error
        derivative = error - self.previous_error[key].get(control_type, 0)
        self.previous_error[key][control_type] = error

        return self.kp * error + self.ki * self.integral[key][control_type] + self.kd * derivative

    def reset(self, key):
        if key in self.integral:
            self.integral[key] = {}
        if key in self.previous_error:
            self.previous_error[key] = {}
