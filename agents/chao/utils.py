import math
import numpy as np
from sturnus.geo import *
from .pid import FlyPid

# def get_turn_delta(plane_value, target_value):
#     temp_value = target_value - plane_value
#     if temp_value < -math.pi:
#         temp_value += math.pi * 2
#     return temp_value

# 画图用
last_target_pitch = 0
last_target_heading = 0
last_target_roll = 0
def fly_point(plane, target, sim_time, data_queue, fly_pid:FlyPid):
    global last_target_pitch, last_target_heading, last_target_roll
    plane_pos = Vector3(
        plane.x, plane.y, plane.z
    )
    target_pos = Vector3(
        target.x, target.y, target.z
    )
    
    # 求目标的姿态角
    to_target_vec = target_pos - plane_pos
    target_yaw = np.arctan2(to_target_vec.y, to_target_vec.x)
    temp = math.sqrt(to_target_vec.x * to_target_vec.x + to_target_vec.y * to_target_vec.y)
    target_pitch = np.arctan2(-to_target_vec.z, temp)
    # 确定转向
    turn_delta = get_turn_delta(plane.yaw, target_yaw)
    temp_turn = math.degrees(turn_delta)
    # 根据差的航向角来计算目标滚转角度
    rate = (abs(temp_turn) / 180) * 1.5
    if rate >= 1:
        rate = 0.99
    if abs(math.degrees(turn_delta)) < 4:
        target_roll = 0
    elif turn_delta > 0:
        target_roll = math.radians(90) * rate
    else:
        target_roll = math.radians(-90) * rate
    
    # 设置目标姿态角角速度
    fly_pid.set_tar_value(target_pitch - plane.pitch, target_yaw - plane.yaw, target_roll - plane.roll)
    cmd_list = fly_pid.get_control_cmd(plane.omega_p, plane.omega_q, plane.omega_r)
    
    # 转弯时降速
    if abs(temp_turn) > 5 and plane.tas > 160:
        cmd_list[3] = 0

    # 限制控制俯仰轴的指令值，防止飞机失控
    if abs(cmd_list[1])  > 0.7:
        cmd_list[1] = 0.7 * np.sign(cmd_list[1])
    
    # 转弯时机头抬高
    if abs(math.degrees(plane.roll)) > 30 and cmd_list[1] > -0.1:
        cmd_list[1] = -0.1
        
    # 飞机倒过来飞时特殊处理，俯仰轴取反方向
    if -90 >= math.degrees(plane.roll) or math.degrees(plane.roll) >= 90: 
        cmd_list[1] = -1 * cmd_list[1]
    # 输出画图数据
    # 时间，真实俯仰，目标俯仰，真实航向，目标航向
    if data_queue != None:
        # 俯仰角
        data_queue.put((sim_time, np.degrees(plane.pitch), np.degrees(last_target_pitch), np.degrees(plane.roll), np.degrees(last_target_roll), cmd_list[0], cmd_list[1], 0, 0))
    last_target_pitch = target_pitch
    last_target_heading = target_yaw
    last_target_roll = target_roll

    return cmd_list

norm_delta_altitude = np.array([500, 0, -500])
norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, -np.pi / 36, 0, np.pi / 36, np.pi / 12, np.pi / 6])
norm_delta_velocity = np.array([0.05, 0, -0.05])
def fly_with_alt_yaw_vel(plane, sim_time, action:list, fly_pid:FlyPid, data_queue=None):
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
    if norm_delta_altitude[action[2]] < 0:
        cmd_list[3] = 0
    elif norm_delta_altitude[action[2]] == 0:
        cmd_list[3] = 0.395 # 匀速
        
    # 限制控制俯仰轴的指令值，防止飞机失控
    if abs(cmd_list[1])  > 0.7:
        cmd_list[1] = 0.7 * np.sign(cmd_list[1])
        
    # 飞机倒过来飞时特殊处理，俯仰轴取反方向
    if -90 >= math.degrees(plane.roll) or math.degrees(plane.roll) >= 90:
        cmd_list[1] = -1 * cmd_list[1]
        
    # 输出画图数据
    # 时间，真实俯仰，目标俯仰，真实航向，目标航向
    if data_queue != None:
        # 俯仰角
        data_queue.put((sim_time, np.degrees(plane.pitch), np.degrees(last_target_pitch), np.degrees(plane.roll), np.degrees(last_target_roll), cmd_list[0], cmd_list[1], 0, 0))
    
    last_target_pitch = target_pitch
    last_target_heading = plane.yaw + math.radians(temp_turn)
    last_target_roll = target_roll
    
    return cmd_list

def degrees_limit(angle):
    if angle > 180:
        return angle - 360
    elif angle < -180:
        return angle + 360
    return angle

class Vector3:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z
        
    def distance(self, point):
        return math.sqrt((self.z - point.z)**2)
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2 +(self.z - point.z)**2)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    

