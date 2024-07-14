import numpy as np
import math
from sturnus.geo import *
from .blue_agent_demo import Agent as BaseAgent

# 选手定义名为Agent的类， 
# 1. 实现 初始化函数(__init__)， 
# 2. 实现 每一帧读取处理态势并返回指令的函数（step)。
class Agent(BaseAgent):
    def __init__(self, side) -> None:
        # 初始化函数
        # 初始化一些相关变量，工具类。
        super().__init__(side)
        self.num_step = 0
        self.history = {}
        self.previous_missile_ind = {}
        self.previous_missile_positions = {}
        self.previous_controls = {}
        self.pid_controller = PIDController()

    def step(self, obs):
        raw_cmd_dict =  super().step(obs)
        self.num_step += 1

        # 机动截胡策略
        # TODO: 应该用脸去贴导弹
        new_cmd_dict = self.control_jiehu(obs, raw_cmd_dict)

        # 发弹截胡策略：
        # TODO：应该追着对面屁股后面发弹，越靠近中轴命中概率越大，只要不是100%，还是妥妥的纯概率事件
        new_cmd_dict = self.weapon_jiehu(obs, new_cmd_dict)
        
        return new_cmd_dict

    def calculate_distance(self, plane_info, missile_info):
        return math.sqrt(
            (missile_info.x - plane_info.x) ** 2 +
            (missile_info.y - plane_info.y) ** 2 +
            (missile_info.z - plane_info.z) ** 2
        )

    def calculate_direction(self, plane_info, missile_info):
        dx = missile_info.x - plane_info.x
        dy = missile_info.y - plane_info.y
        dz = missile_info.z - plane_info.z
        horizontal_distance = math.sqrt(dx ** 2 + dy ** 2)
        azimuth = math.atan2(dy, dx)
        elevation = math.atan2(dz, horizontal_distance)
        return azimuth, elevation

    def estimate_missile_velocity(self, missile_info, key):
        if key in self.previous_missile_positions:
            prev_x, prev_y, prev_z = self.previous_missile_positions[key]
            dt = 1 / 20.0  # 每个step的间隔是1/20秒
            v_x = (missile_info.x - prev_x) / dt
            v_y = (missile_info.y - prev_y) / dt
            v_z = (missile_info.z - prev_z) / dt
        else:
            v_x, v_y, v_z = 0, 0, 0
        self.previous_missile_positions[key] = (missile_info.x, missile_info.y, missile_info.z)
        return v_x, v_y, v_z

    def predict_missile_position(self, missile_info, v_x, v_y, v_z, steps_ahead=10):
        predicted_x = missile_info.x + v_x * steps_ahead * (1 / 20.0)
        predicted_y = missile_info.y + v_y * steps_ahead * (1 / 20.0)
        predicted_z = missile_info.z + v_z * steps_ahead * (1 / 20.0)
        return predicted_x, predicted_y, predicted_z

    def calculate_control(self, plane_info, missile_info, azimuth, elevation, key):
        target_heading = math.degrees(azimuth)
        target_pitch = math.degrees(elevation)

        current_heading = plane_info.yaw
        current_pitch = plane_info.pitch

        heading_diff = (target_heading - current_heading + 180) % 360 - 180
        pitch_diff = target_pitch - current_pitch

        aileron = self.pid_controller.update(heading_diff, key, 'aileron')
        elevator = self.pid_controller.update(pitch_diff, key, 'elevator')
        rudder = aileron
        throttle = 1.0

        prev_aileron, prev_elevator, prev_rudder, prev_throttle = self.previous_controls.get(key, [0, 0, 0, 1])
        smooth_factor = 0.1
        aileron = prev_aileron * (1 - smooth_factor) + aileron * smooth_factor
        elevator = prev_elevator * (1 - smooth_factor) + elevator * smooth_factor
        rudder = prev_rudder * (1 - smooth_factor) + rudder * smooth_factor
        throttle = prev_throttle * (1 - smooth_factor) + throttle * smooth_factor

        self.previous_controls[key] = [aileron, elevator, rudder, throttle]

        return [aileron, elevator, rudder, throttle]

    def control_jiehu(self, obs, raw_cmd_dict):
        my_plane_id_list = list(obs.my_planes.keys())
        enemy_plane_id_list = list(obs.enemy_planes.keys())

        if not enemy_plane_id_list:
            return raw_cmd_dict

        for key, value in raw_cmd_dict.items():
            my_plane_info = obs.my_planes[key]
            min_distance = float('inf')
            closest_missile = None

            for entity_info in obs.rws_infos:
                if entity_info.ind in enemy_plane_id_list:
                    continue

                if key in entity_info.alarm_ind_list:
                    distance = self.calculate_distance(my_plane_info, entity_info)
                    if distance < min_distance:
                        min_distance = distance
                        closest_missile = entity_info

            if closest_missile:
                if key in self.previous_missile_ind and self.previous_missile_ind[key] != closest_missile.ind:
                    self.history[key] = []
                self.previous_missile_ind[key] = closest_missile.ind

                v_x, v_y, v_z = self.estimate_missile_velocity(closest_missile, key)
                predicted_x, predicted_y, predicted_z = self.predict_missile_position(closest_missile, v_x, v_y, v_z)
                azimuth, elevation = self.calculate_direction(my_plane_info, closest_missile)
                control = self.calculate_control(my_plane_info, closest_missile, azimuth, elevation, key)
                value['control'] = control  # 覆盖原有的控制指令

                self.history.setdefault(key, []).append({
                    'step': self.num_step,
                    'plane_info': my_plane_info,
                    'missile_info': closest_missile,
                    'control': control
                })

                if self.num_step % 10 == 0:
                    print(f"Step: {self.num_step}")
                    print(f"Plane ID: {key}, My plane coordinates: (x: {my_plane_info.x}, y: {my_plane_info.y}, z: {my_plane_info.z}, yaw: {my_plane_info.yaw}, v_north: {my_plane_info.v_north}, v_east: {my_plane_info.v_east}, v_down: {my_plane_info.v_down})")
                    print(f"Closest missile ind: {closest_missile.ind}, Coordinates: (x: {closest_missile.x}, y: {closest_missile.y}, z: {closest_missile.z})")
                    print(f"Control: aileron={control[0]}, elevator={control[1]}, rudder={control[2]}, throttle={control[3]}")

        return raw_cmd_dict  # 返回覆盖后的字典

    # 目前比较简单，就是晚一点发弹，一开始避免对喷
    def weapon_jiehu(self, obs, raw_cmd_dict):
        enemy_plane_id_list = list(obs.enemy_planes.keys())

        for key, value in raw_cmd_dict.items():
            if 'weapon' in value and self.num_step < 5000:
                del value['weapon']
        
        return raw_cmd_dict


class PIDController:
    def __init__(self, kp=0.1, ki=0.01, kd=0.05):
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