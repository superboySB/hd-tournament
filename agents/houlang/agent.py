import numpy as np
import math
from sturnus.geo import *
from .funcs import FlyPid  # 确保 FlyPid 模块正确引用

last_target_pitch = 0
last_target_heading = 0
last_target_roll = 0
norm_delta_altitude = np.array([500, 0, -500])
norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, -np.pi / 36, 0, np.pi / 36, np.pi / 12, np.pi / 6])
norm_delta_velocity = np.array([0.05, 0, -0.05])

class Agent:
    def __init__(self, side):
        self.side = side
        self.id_pidctl_dict = {}
        self.ini_pid = False
        self.waypoints = []
        self.run_counts = 0
        self.phase = 1  # 1 for approach, 2 for circling
        self.heat_zone_center = Vector3(0, 0, -3000)
        self.heat_zone_radius = 15000
        self.current_enemy_plane_id_list = []
    
    def calculate_distance(self, plane_info, missile_info):
        return math.sqrt(
            (missile_info.x - plane_info.x) ** 2 +
            (missile_info.y - plane_info.y) ** 2 +
            (missile_info.z - plane_info.z) ** 2
        )
    
    def get_action_cmd(self, target, plane, mode="fix_point"):
        """_summary_

        Args:
            target (list/myplaneinfo): 目标信息
            plane (myplaneinfo): 自身信息 
            mode (str): fix_point / plane
        Returns:
            list: [delta_alt, delta_yaw(角度), delta_vel]
        """
        
        plane_pos = Vector3(
            plane.x, plane.y, plane.z
        )
        target_pos = Vector3(
            target.x, target.y, target.z
        )
        action = np.zeros(3, dtype=int)
        if target.z - plane.z < -500:# 升高
            action[0] = 0
            print("升高", end=' ')
        elif target.z - plane.z > 500:# 降高
            action[0] = 2
            print("降高", end=' ')
        else:
            action[0] = 1# 保持
            print("保持", end=' ')
        
        to_target_vec = target_pos - plane_pos
        target_yaw = np.arctan2(to_target_vec.y, to_target_vec.x)
        delta_yaw = degrees_limit(math.degrees(target_yaw - plane.yaw))
        
        if delta_yaw > 30:# 右转 30度
            action[1] = 6
            print("右转 30度", end=' ')
        elif 15 < delta_yaw:# 右转 15度
            action[1] = 5
            print("右转 15度", end=' ')
        elif 5 < delta_yaw:# 右转 5度
            action[1] = 4
            print("右转 5度", end=' ')
        elif -5 < delta_yaw:# 左转 5度
            action[1] = 3
            print("直走", end=' ')
        elif -15 < delta_yaw:# 直走
            action[1] = 2
            print("左转 5度", end=' ')
        elif -30 < delta_yaw:# 左转 15度
            action[1] = 1
            print("左转 15度", end=' ')
        else:
            action[1] = 0# 左转30度
            print("左转 30度", end=' ')
        
        if mode == "plane":
            if target.tas - plane.tas > 10:
                action[2] = 0
                print("加速", end=' ')
            elif target.tas - plane.tas < -10:
                action[2] = 2
                print("减速", end=' ')
            else:
                action[2] = 1
                print("匀速", end=' ')
        else:
            action[2] = 0
            print("匀速", end=' ')
            
        return action

    def step(self, obs):
        enemy_plane_id_list = list(obs.enemy_planes.keys())
        if enemy_plane_id_list:
            self.current_enemy_plane_id_list = enemy_plane_id_list

        cmd_dict = {}
        for my_id, my_plane in obs.my_planes.items():
            if not self.ini_pid or my_id not in self.id_pidctl_dict:
                self.id_pidctl_dict[my_id] = FlyPid()

            closest_missile = None
            min_distance = 100e4
            self.phase = 1
            for entity_info in obs.rws_infos:
                if entity_info.ind in self.current_enemy_plane_id_list:
                    continue
                if my_id in entity_info.alarm_ind_list:
                    distance = self.calculate_distance(my_plane, entity_info)
                    if distance < min_distance:
                        min_distance = distance
                        closest_missile = entity_info
            if min_distance < 20000:
                self.phase = 2
                
            # 情况2：占领热区
            if self.phase == 1:
                print("占领热区", end=' ')
                target_pos = self.heat_zone_center
                action = self.get_action_cmd(target_pos, my_plane, "fix_point")
            elif self.phase == 2:
                print("躲避missile!!!", end=' ')
                target_pos = Vector3(closest_missile.x, closest_missile.y, closest_missile.z)
                action = [1,6,2]
            else:
                raise NotImplementedError
                
            cmd = {'control': fly_with_alt_yaw_vel(my_plane, action, self.id_pidctl_dict[my_id])}
            cmd_dict[my_id] = cmd

            print("Step: ",self.run_counts,", ID: ", my_id, ", 位置：", [my_plane.x,my_plane.y,my_plane.z], "目标：", [target_pos.x,target_pos.y,target_pos.z],"控制:", cmd["control"])

        self.run_counts += 1
        return cmd_dict

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance(self, point):
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2 + (self.z - point.z)**2)

def degrees_limit(angle):
    if angle > 180:
        return angle - 360
    elif angle < -180:
        return angle + 360
    return angle

def fly_with_alt_yaw_vel(plane, action:list, fly_pid:FlyPid):
    """_summary_

    Args:
        action (list): [(0-2), (0-4), (0-2)] ,

    Returns:
        list: [aileron, elevator, rudder, throttle]
    """
    global last_target_pitch, last_target_heading, last_target_roll
    # 确定转向
    temp_turn = math.degrees(norm_delta_heading[action[1]])
    # 根据当前状态想移动的角度来计算目标滚转角度
    rate = (abs(temp_turn) / 35)
    # if rate >= 1:
    #     rate = 0.99
    if abs(temp_turn) < 4:
        target_roll = 0
    elif temp_turn > 0: 
        target_roll = math.radians(90) * rate # 右转
    else:
        target_roll = math.radians(-90) * rate # 左转
    
    target_pitch = np.arctan2(norm_delta_altitude[action[0]], 500)
    
    # 设置目标姿态角角速度
    fly_pid.set_tar_value(target_pitch - plane.pitch, norm_delta_heading[action[1]], target_roll - plane.roll)
    cmd_list = fly_pid.get_control_cmd(plane.omega_p, plane.omega_q, plane.omega_r)
    
    # 控制加力来改变速度
    if norm_delta_velocity[action[2]] < 0:
        cmd_list[3] = 0
    elif norm_delta_velocity[action[2]] == 0:
        cmd_list[3] = 0.395 # 匀速
        
    # 限制控制俯仰轴的指令值，防止飞机失控
    if abs(cmd_list[1])  > 0.7:
        cmd_list[1] = 0.7 * np.sign(cmd_list[1])
        
    # 飞机倒过来飞时特殊处理，俯仰轴取反方向
    if -90 >= math.degrees(plane.roll) or math.degrees(plane.roll) >= 90:
        cmd_list[1] = -1 * cmd_list[1]
    
    last_target_pitch = target_pitch
    last_target_heading = plane.yaw + math.radians(temp_turn)
    last_target_roll = target_roll
    
    return cmd_list
