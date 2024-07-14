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

    def step(self, obs):
        raw_cmd_dict =  super().step(obs)
        self.num_step += 1
        # print(raw_cmd_dict)

        # 机动截胡策略
        new_cmd_dict = self.control_jiehu(obs, raw_cmd_dict)

        # 发弹截胡策略：
        new_cmd_dict = self.weapon_jiehu(obs, new_cmd_dict)
        
        return new_cmd_dict

    # TODO：尝试添加用脸接弹的机动策略
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
                    # 计算距离
                    distance = math.sqrt(
                        (entity_info.x - my_plane_info.x) ** 2 +
                        (entity_info.y - my_plane_info.y) ** 2 +
                        (entity_info.z - my_plane_info.z) ** 2
                    )
                    
                    # 找到最近的导弹
                    if distance < min_distance:
                        min_distance = distance
                        closest_missile = entity_info
            
            if closest_missile and self.num_step % 10 ==0:
                print(f"Plane ID: {key}, My plane coordinates: (x: {my_plane_info.x}, y: {my_plane_info.y}, z: {my_plane_info.z})")
                print(f"Closest missile ind: {closest_missile.ind}, Coordinates: (x: {closest_missile.x}, y: {closest_missile.y}, z: {closest_missile.z})")

        return raw_cmd_dict

    # 目前比较简单，就是晚一点发弹，一开始避免对喷
    # TODO：应该追着对面屁股后面发弹，越靠近中轴命中概率越大，只要不是100%，就妥妥的纯概率事件
    def weapon_jiehu(self, obs, raw_cmd_dict):
        for key, value in raw_cmd_dict.items():
            if 'weapon' in value and self.num_step < 3000:
                del value['weapon']
        
        return raw_cmd_dict