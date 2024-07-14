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

    def step(self, obs):
        raw_cmd_dict =  super().step(obs)
        self.num_step += 1

        # 机动截胡策略
        # TODO: 应该用脸去贴导弹
        new_cmd_dict = self.control_jiehu(obs, raw_cmd_dict)

        # 发弹截胡策略：
        # TODO：应该追着对面屁股后面发弹，越靠近中轴命中概率越大，只要不是100%，就妥妥的纯概率事件
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

    def calculate_control(self, plane_info, missile_info, azimuth, elevation):
        # 计算需要调整的控制量
        target_heading = math.degrees(azimuth)
        target_pitch = math.degrees(elevation)

        # 计算飞机需要旋转的角度
        current_heading = plane_info.yaw
        heading_diff = target_heading - current_heading

        # 简化版的控制策略
        aileron = math.sin(math.radians(heading_diff))
        elevator = -math.sin(math.radians(target_pitch))
        rudder = aileron  # 简单起见，假设偏航控制和横滚控制一致
        throttle = 1  # 全速前进

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
                
                azimuth, elevation = self.calculate_direction(my_plane_info, closest_missile)
                control = self.calculate_control(my_plane_info, closest_missile, azimuth, elevation)
                value['control'] = control  # 覆盖原有的控制指令

                self.history.setdefault(key, []).append({
                    'step': self.num_step,
                    'plane_info': my_plane_info,
                    'missile_info': closest_missile,
                    'control': control
                })

                if self.num_step % 10 == 0:
                    print(f"Plane ID: {key}, My plane coordinates: (x: {my_plane_info.x}, y: {my_plane_info.y}, z: {my_plane_info.z})")
                    print(f"Closest missile ind: {closest_missile.ind}, Coordinates: (x: {closest_missile.x}, y: {closest_missile.y}, z: {closest_missile.z})")
                    print(f"Control: aileron={control[0]}, elevator={control[1]}, rudder={control[2]}, throttle={control[3]}")

        return raw_cmd_dict  # 返回覆盖后的字典

    # 目前比较简单，就是晚一点发弹，一开始避免对喷
    def weapon_jiehu(self, obs, raw_cmd_dict):
        for key, value in raw_cmd_dict.items():
            if 'weapon' in value and self.num_step < 5000:
                del value['weapon']
        
        return raw_cmd_dict