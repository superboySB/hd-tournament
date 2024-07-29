import numpy as np
import math
from .funcs import FlyPid  # 确保 FlyPid 模块正确引用

last_target_pitch = 0
last_target_heading = 0
last_target_roll = 0
norm_delta_altitude = np.array([500, 0, -500])
norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, -np.pi / 36, 0, np.pi / 36, np.pi / 12, np.pi / 6])
norm_delta_velocity = np.array([0.05, 0, -0.05])

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

class Agent:
    def __init__(self, side):
        self.side = side
        self.id_pidctl_dict = {}
        self.ini_pid = False
        self.waypoints = []
        self.phase = 1  # 1 for approach, 2 for circling
        self.heat_zone_center = Vector3(0, 0, -2000)
        self.heat_zone_radius = 15000
        self.generate_waypoints()  # 初始化路径点

    def generate_waypoints(self):
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        self.waypoints = [Vector3(self.heat_zone_center.x + 5000 * np.cos(angle), self.heat_zone_center.y + 5000 * np.sin(angle), -1000) for angle in angles]

    def find_farthest_waypoint(self, plane_pos):
        max_distance = 0
        farthest_index = 0
        for i, point in enumerate(self.waypoints):
            distance = plane_pos.distance(point)
            if distance > max_distance:
                max_distance = distance
                farthest_index = i
        return farthest_index

    def get_action_cmd(self, target_pos, plane):
        """从之前的代码生成，未修改."""
        plane_pos = Vector3(plane.x, plane.y, plane.z)
        delta_pos = target_pos - plane_pos

        action = np.zeros(3, dtype=int)
        if target_pos.z - plane.z < -500:  # Ascend
            action[0] = 0
        elif target_pos.z - plane.z > 500:  # Descend
            action[0] = 2
        else:
            action[0] = 1  # Maintain altitude

        delta_yaw = degrees_limit(math.degrees(np.arctan2(delta_pos.y, delta_pos.x) - plane.yaw))
        # Determine the yaw adjustment needed
        if delta_yaw > 30:
            action[1] = 6
        elif 15 < delta_yaw <= 30:
            action[1] = 5
        elif 5 < delta_yaw <= 15:
            action[1] = 4
        elif -5 < delta_yaw <= 5:
            action[1] = 3
        elif -15 < delta_yaw <= -5:
            action[1] = 2
        elif -30 < delta_yaw <= -15:
            action[1] = 1
        else:
            action[1] = 0  # Turn hard left

        action[2] = 0  # Follow speed setting from original code, not dynamic

        return action

    def step(self, obs):
        cmd_dict = {}
        for id, plane in obs.my_planes.items():
            plane_pos = Vector3(plane.x, plane.y, plane.z)
            if not self.ini_pid or id not in self.id_pidctl_dict:
                self.id_pidctl_dict[id] = FlyPid()

            if self.phase == 1:
                distance_to_center = plane_pos.distance(self.heat_zone_center)
                if distance_to_center <= self.heat_zone_radius:
                    self.phase = 2
                    self.generate_waypoints()

            if self.phase == 1:
                target_pos = self.heat_zone_center
            elif self.run_counts % 500 == 0:  # 每500步检查最远的路径点
                farthest_index = self.find_farthest_waypoint(plane_pos)
                target_pos = self.waypoints[farthest_index]

            action = self.get_action_cmd(target_pos, plane)
            cmd = {'control': fly_with_alt_yaw_vel(plane, action, self.id_pidctl_dict[id])}
            cmd_dict[id] = cmd

            if self.run_counts % 10 == 0:
                print(f"Step: {obs.step_count}, ID: {id}, Phase: {self.phase}, Target: ({target_pos.x}, {target_pos.y}, {target_pos.z}), Control: {cmd['control']}")

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
