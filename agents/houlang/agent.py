import numpy as np
import math
from .funcs import FlyPid

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def distance(self, point):
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2 + (self.z - point.z)**2)

class Agent:
    def __init__(self, side):
        self.side = side
        self.cmd_id = 0
        self.run_counts = 0
        self.tar_id = 0
        self.ini_pid = False
        self.id_pidctl_dict = {}
        self.waypoints = []
        self.phase = 1  # 1 for approach, 2 for circling
        self.heat_zone_center = Vector3(0, 0, -2000)
        self.heat_zone_radius = 15000

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

    def get_action_cmd(self, plane_pos, target_pos):
        delta_pos = target_pos - plane_pos
        action = [
            delta_pos.z,  # Altitude difference
            math.degrees(np.arctan2(delta_pos.y, delta_pos.x)),  # Yaw difference
            np.linalg.norm([delta_pos.x, delta_pos.y, delta_pos.z])  # Distance as a proxy for speed adjustment
        ]
        return action

    def step(self, obs):
        self.run_counts += 1
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
            else:
                if self.run_counts % 1000 == 0:  # Check for the farthest waypoint every 1000 steps
                    farthest_index = self.find_farthest_waypoint(plane_pos)
                    target_pos = self.waypoints[farthest_index]

            action = self.get_action_cmd(plane_pos, target_pos)
            cmd = {'control': fly_with_alt_yaw_vel(plane, action, self.id_pidctl_dict[id])}
            cmd_dict[id] = cmd

            if self.run_counts % 10 == 0:
                print(f"Step: {self.run_counts}, ID: {id}, Phase: {self.phase}, Target: ({target_pos.x}, {target_pos.y}, {target_pos.z})")

        return cmd_dict

def fly_with_alt_yaw_vel(plane, action, fly_pid):
    # Calculate target pitch and roll based on the action
    target_pitch = np.arctan2(action[0], 500)  # Simplified calculation
    target_roll = math.radians(90) * (abs(action[1]) / 35)
    target_roll = target_roll if action[1] > 0 else -target_roll
    
    # Set target attitude rate of change
    fly_pid.set_tar_value(target_pitch - plane.pitch, action[1], target_roll - plane.roll)
    cmd_list = fly_pid.get_control_cmd(plane.omega_p, plane.omega_q, plane.omega_r)

    # Throttle adjustments based on desired speed change
    if action[2] < 0:
        cmd_list[3] = 0
    elif action[2] == 0:
        cmd_list[3] = 0.395
    else:
        cmd_list[3] = 1  # Maximum throttle
    
    return cmd_list
