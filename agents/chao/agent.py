import numpy as np
import math
from sturnus.geo import *
from .pid import FlyPid
# 选手定义名为Agent的类， 
# 1. 实现 初始化函数(__init__)， 
# 2. 实现 每一帧读取处理态势并返回指令的函数（step)。

target_info = 0
last_target_pitch = 0
last_target_heading = 0
last_target_roll = 0
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
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
class Agent:
    def __init__(self, side) -> None:
        # 初始化函数
        # 初始化一些相关变量，工具类。
        self.side = side
        self.cmd_id = 0
        self.run_counts = 0
        self.tar_id = 0
        self.ini_pid = False
        self.id_pidctl_dict = {}

    def weapon_cmd(self, type:str, target:int) -> dict:
        return {"type": type, "target": target}
    
    def get_action_cmd(self, target, plane):
        """_summary_

        Args:
            target (_type_): _description_

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
        if target.height - plane.height > 500:# 升高
            action[0] = 0
            print("升高", end=' ')
        elif target.height - plane.height < -500:# 降高
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
            
        if target.tas - plane.tas > 10:# 加速跟随
            action[2] = 0
            print("加速", end=' ')
        elif target.tas - plane.tas < -10:# 尽量匀速
            action[2] = 2
            print("减速", end=' ')
        else:
            action[2] = 1# 减速
            print("匀速", end=' ')
            
        return action
    
    def step(self, obs, blue_obs, data_queue=None):
        cmd_dict = {}
        global target_point, target_info
        self.run_counts += 1
        # 获取此步长的地方信息
        for id, tar_plane in blue_obs.my_planes.items():
            if self.tar_id == 0:
                self.tar_id = tar_plane.ind
                print("目标：", self.tar_id)
                target_info = tar_plane
                break
            elif self.tar_id == id:
                target_info = tar_plane
                break
        
        if self.tar_id:
            for id, plane in obs.my_planes.items():
                if not self.ini_pid:
                    self.id_pidctl_dict[id] = FlyPid()
                cmd = {}
                if self.cmd_id == 0 and id != self.tar_id:
                    print("控制：", id)
                    self.cmd_id = id
                if id == self.cmd_id:
                    action = self.get_action_cmd(target_info, plane)  # 替代了一个high-level来写规则了
                    cmd['control'] = fly_with_alt_yaw_vel(plane, obs.sim_time, action, fly_pid=self.id_pidctl_dict[id], data_queue=data_queue)
                    print(f"{self.cmd_id} to {self.tar_id} action: {action}, cmd: {cmd['control']}, omega_r: {math.degrees(plane.omega_r)}")
                    
                    cmd_dict[id] = cmd
            self.ini_pid = True
        return cmd_dict