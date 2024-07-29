import numpy as np
import math
from sturnus.geo import *
from .pid import FlyPid

# 全局变量，目标坐标
target_coordinates = [0, 0, -2000]
target_info = 0
last_target_pitch = 0
last_target_heading = 0
last_target_roll = 0
norm_delta_altitude = np.array([500, 0, -500])
norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, -np.pi / 36, 0, np.pi / 36, np.pi / 12, np.pi / 6])
norm_delta_velocity = np.array([0.05, 0, -0.05])

class Vector3:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z
        
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
def degrees_limit(angle):
    if angle > 180:
        return angle - 360
    elif angle < -180:
        return angle + 360
    return angle

class Agent:
    def __init__(self, side) -> None:
        self.side = side
        self.id_pidctl_dict = {}
        self.ini_pid = False

    def get_action_cmd(self, target_pos, plane):
        plane_pos = Vector3(plane.x, plane.y, plane.z)
        delta_pos = target_pos - plane_pos
        
        action = np.zeros(3, dtype=int)
        if target_pos.z - plane.z < -500:# 升高
            action[0] = 0
            print("升高", end=' ')
        elif target_pos.z - plane.z > 500:# 降高
            action[0] = 2
            print("降高", end=' ')
        else:
            action[0] = 1# 保持
            print("保持", end=' ')

        delta_yaw = degrees_limit(math.degrees(np.arctan2(delta_pos.y, delta_pos.x) - plane.yaw))

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
            
        # 加速跟随
        action[2] = 0

        return action

    def step(self, obs):
        cmd_dict = {}
        global target_coordinates
        target_pos = Vector3(*target_coordinates)

        for id, plane in obs.my_planes.items():
            if id not in self.id_pidctl_dict:
                self.id_pidctl_dict[id] = FlyPid()
            
            print("ID: ", id, "位置：", [plane.x,plane.y,plane.z], "目标：", target_coordinates)

            action = self.get_action_cmd(target_pos, plane)
            cmd = {
                'control': fly_with_alt_yaw_vel(plane, action, self.id_pidctl_dict[id])
            }
            cmd_dict[id] = cmd
        return cmd_dict

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
    
    last_target_pitch = target_pitch
    last_target_heading = plane.yaw + math.radians(temp_turn)
    last_target_roll = target_roll
    
    return cmd_list
