import numpy as np
import math
from sturnus.geo import *
import warnings
from .blue_agent_demo import Agent as BaseAgent
from .funcs import FlyPid, Vector3, degrees_limit, is_facing_target  # 确保 FlyPid 模块正确引用


last_target_pitch = 0
last_target_heading = 0
last_target_roll = 0
norm_delta_altitude = np.array([500, 0, -500])
norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, -np.pi / 36, 0, np.pi / 36, np.pi / 12, np.pi / 6])
norm_delta_velocity = np.array([0.05, 0, -0.05])

class Agent(BaseAgent):
    def __init__(self, side):
        super().__init__(side)
        self.id_pidctl_dict = {}
        self.ini_pid = False
        self.run_counts = 0
        self.phase = 1  # 1 for approach, 2 for circling
        self.heat_zone_center = Vector3(0, 0, -3000) # TODO: 改为建立保护有人机为主的假热区中心，真热区：Vector3(0, 0, -3000)
        self.heat_zone_radius = 15000

        self.fake_head_zone_center_dict = {
            "av1": Vector3(55e3, 0, -4000),
            "av2": Vector3(-55e3, 0, 5000),
            "uav1": Vector3(40e3, 0, -3000),
            "uav2": Vector3(-40e3, 0, 4000),
            "uav3": Vector3(25e3, 0, -2000),
            "uav4": Vector3(-25e3, 0, 3000),
        }
        self.assigned_targets = {}  # 存储每个飞机分配的目标点
        self.use_fake_heat_zone = {}  # 每个飞机是否使用假热区

        self.full_enemy_plane_id_list = []
        self.myplane_tracks = {}  # 记录我方每架飞机的轨迹
        self.enemy_plane_tracks = {} # 记录敌方每架飞机的轨迹
        self.missile_tracks = {}  # 记录每个导弹的轨迹
        self.missile_plane_distance_tracks = {}

        self.expired_missiles = set()  # 存储过时的导弹 ID
        self.dangerous_missiles = set()

        # 初始化一些策略状态， 统计变量。
        self.missile_cds = {}
        self.mid_missile_time = {}    # 记录中距弹上一次打弹时间，用于设置打弹cd
        self.short_missile_time = {}    # 记录近距弹上一次打弹时间，用于设置打弹cd
        self.mid_lock_time = 0
    
    def calculate_distance_2d(self, plane_info, missile_info):
        return math.sqrt(
            (missile_info.x - plane_info.x) ** 2 +
            (missile_info.y - plane_info.y) ** 2
        )
    
    def calculate_distance_3d(self, plane_info, missile_info):
        return math.sqrt(
            (missile_info.x - plane_info.x) ** 2 +
            (missile_info.y - plane_info.y) ** 2 +
            (missile_info.y - plane_info.z) ** 2
        )
    
    def assign_targets(self, obs, debug=False):
        # 初始化目标分配
        human_plane_ids = [my_id for my_id, my_plane in obs.my_planes.items() if not my_plane.is_uav]
        uav_plane_ids = [my_id for my_id, my_plane in obs.my_planes.items() if my_plane.is_uav]

        # 提取 fake_heat_zone 中的目标点
        human_targets = ["av1", "av2"]
        uav_targets = ["uav1", "uav2", "uav3", "uav4"]

        # 分配给有人机
        for my_id in human_plane_ids:
            closest_target = min(human_targets, key=lambda t: self.calculate_distance_3d(obs.my_planes[my_id], self.fake_head_zone_center_dict[t]))
            self.assigned_targets[my_id] = self.fake_head_zone_center_dict[closest_target]
            self.use_fake_heat_zone[my_id] = True
            human_targets.remove(closest_target)

        # 分配给无人机
        for my_id in uav_plane_ids:
            closest_target = min(uav_targets, key=lambda t: self.calculate_distance_3d(obs.my_planes[my_id], self.fake_head_zone_center_dict[t]))
            self.assigned_targets[my_id] = self.fake_head_zone_center_dict[closest_target]
            self.use_fake_heat_zone[my_id] = True
            uav_targets.remove(closest_target)

        # 打印分配结果
        if debug:
            print("目标分配结果:")
            for my_id, target in self.assigned_targets.items():
                print(f"飞机ID: {my_id}, 目标: {[target.x,target.y,target.z]}")

    def get_action_cmd(self, target, plane, mode="fix_point", debug=False):
        """_summary_

        Args:
            target (list/myplaneinfo): 目标信息
            plane (myplaneinfo): 自身信息 
            mode (str): fix_point / plane / missile
        Returns:
            list: [delta_alt, delta_yaw(角度), delta_vel]
        """
        if mode == "fix_point":
            if debug:
                print("占领热区", end=' ')
        elif mode == "missile":
            if debug:
                print("躲避missile", end=' ')
        elif mode == "plane":
            if debug:
                print("狗斗plane", end=' ')
        else:
            raise NotImplementedError
        
        plane_pos = Vector3(
            plane.x, plane.y, plane.z
        )
        target_pos = Vector3(
            target.x, target.y, target.z
        )
            
        action = np.zeros(3, dtype=int)

        # --------------------------------------------------------------------
        if target_pos.z - plane.z < -300:# 升高
            action[0] = 0
            if debug:
                print("升高", end=' ')
        elif target_pos.z - plane.z > 300:# 降高
            action[0] = 2
            if debug:
                print("降高", end=' ')
        else:
            action[0] = 1   # 保持
            if debug:
                print("保持", end=' ')

        # --------------------------------------------------------------------
        to_target_vec = target_pos - plane_pos
        target_yaw = np.arctan2(to_target_vec.y, to_target_vec.x)
        delta_yaw = degrees_limit(math.degrees(target_yaw - plane.yaw))
        
        if delta_yaw > 30:# 右转 30度
            action[1] = 6
            if debug:
                print("右转 30度", end=' ')
        elif 15 < delta_yaw:# 右转 15度
            action[1] = 5
            if debug:
                print("右转 15度", end=' ')
        elif 5 < delta_yaw:# 右转 5度
            action[1] = 4
            if debug:
                print("右转 5度", end=' ')
        elif -5 < delta_yaw:# 左转 5度
            action[1] = 3
            if debug:
                print("直走", end=' ')
        elif -15 < delta_yaw:# 直走
            action[1] = 2
            if debug:
                print("左转 5度", end=' ')
        elif -30 < delta_yaw:# 左转 15度
            action[1] = 1
            if debug:
                print("左转 15度", end=' ')
        else:
            action[1] = 0# 左转30度
            if debug:
                print("左转 30度", end=' ')

        # --------------------------------------------------------------------
        action[2] = 0    # 1是匀速，2是减速
        if debug:
            print("加速", end=' ')
            
        return action

    def get_weapon_launch_info(self, obs, my_plane, debug=False):
        weapon_launch_info = {}
        if len(my_plane.mid_lock_list)>0 and my_plane.loadout.get('mid_missile', 0)>0:
                best_target = None
                min_distance = float('inf')
                for target_id in my_plane.mid_lock_list:
                    if target_id in self.enemy_plane_tracks:
                        target_positions = np.array(self.enemy_plane_tracks[target_id][-20:])
                        aircraft_positions = np.array(self.myplane_tracks[my_plane.ind][-20:])
                        _, cos_theta, _, _ = is_facing_target(target_positions, aircraft_positions, debug=debug)
                        
                        if cos_theta > 0.3:
                            distance = self.calculate_distance_3d(Vector3(*aircraft_positions[-1]), Vector3(*target_positions[-1]))
                            if distance < min_distance and distance < 30000:
                                min_distance = distance
                                best_target = target_id

                if best_target:
                    weapon_launch_info = {
                        'type': 'mid_missile',
                        'target': best_target
                    }
                    self.mid_missile_time[my_plane.ind] = obs.sim_time

        if len(my_plane.short_lock_list)>0 and my_plane.loadout.get('short_missile', 0)>0:
                best_target = None
                min_distance = float('inf')
                for target_id in my_plane.short_lock_list:
                    if target_id in self.enemy_plane_tracks:
                        target_positions = np.array(self.enemy_plane_tracks[target_id][-20:])
                        aircraft_positions = np.array(self.myplane_tracks[my_plane.ind][-20:])
                        _, cos_theta, _, _ = is_facing_target(target_positions, aircraft_positions,debug=debug)
                        
                        if cos_theta > -0.3:
                            distance = self.calculate_distance_3d(Vector3(*aircraft_positions[-1]), Vector3(*target_positions[-1]))
                            if distance < min_distance:
                                min_distance = distance
                                best_target = target_id

                if best_target:
                    weapon_launch_info = {
                        'type': 'short_missile',
                        'target': best_target
                    }
                    self.short_missile_time[my_plane.ind] = obs.sim_time

        return weapon_launch_info

    def update_enemy_plane_tracks(self, obs):
        if len(obs.awacs_infos) and len(self.full_enemy_plane_id_list)==0:
            for awacs_i in obs.awacs_infos:
                self.full_enemy_plane_id_list.append(awacs_i.ind)
                self.enemy_plane_tracks[awacs_i.ind] = []
        
        if len(obs.awacs_infos):
            alive_enemy_inds = [awacs_i.ind for awacs_i in obs.awacs_infos]
            for enemy_id in self.full_enemy_plane_id_list:
                if enemy_id not in alive_enemy_inds:
                    # 如果敌机不再活跃，从跟踪列表中移除该敌机的跟踪信息
                    if enemy_id in self.enemy_plane_tracks:
                        del self.enemy_plane_tracks[enemy_id]
                    # 同时从全敌机列表中移除该敌机
                    self.full_enemy_plane_id_list.remove(enemy_id)

            percise_visible_enemy_ids = set()
            for enemy_id, enemy_plane in obs.enemy_planes.items():
                self.enemy_plane_tracks[enemy_id].append([enemy_plane.x, enemy_plane.y, enemy_plane.z])
                percise_visible_enemy_ids.add(enemy_id)

            rws_visible_enemy_ids = set()
            for entity_info in obs.rws_infos:
                if entity_info.ind in self.full_enemy_plane_id_list and entity_info.ind not in percise_visible_enemy_ids:
                    self.enemy_plane_tracks[entity_info.ind].append([entity_info.x, entity_info.y, entity_info.z])
                    rws_visible_enemy_ids.add(entity_info.ind)

            for entity_info in obs.awacs_infos:
                if entity_info.ind in self.full_enemy_plane_id_list and entity_info.ind not in percise_visible_enemy_ids and \
                    entity_info.ind not in rws_visible_enemy_ids:
                    self.enemy_plane_tracks[entity_info.ind].append([entity_info.x, entity_info.y, entity_info.z])
                
            max_length = max((len(tracks) for tracks in self.enemy_plane_tracks.values()), default=0)
            # print("max_length: ", max_length)
            for enemy_id, tracks in self.enemy_plane_tracks.items():
                if len(tracks) < max_length:
                    warnings.warn("各存活敌机的位置记录不可能时间戳对不齐", UserWarning)
                

    def step(self, obs):
        debug_flag = False
        # if self.run_counts % 100 == 0:
        #     print("\n------------------------------------------------------------------\n")
        #     debug_flag = True

        if self.run_counts == 0:
            self.assign_targets(obs,debug=debug_flag)

        self.update_enemy_plane_tracks(obs)
        raw_cmd_dict =  super().step(obs)

        for my_id, my_plane in obs.my_planes.items():
            if not self.ini_pid or my_id not in self.id_pidctl_dict:
                self.id_pidctl_dict[my_id] = FlyPid()

            if my_id not in self.myplane_tracks:
                self.myplane_tracks[my_id] = []
            self.myplane_tracks[my_id].append([my_plane.x, my_plane.y, my_plane.z])

            closest_missile = None
            if self.use_fake_heat_zone[my_id]:
                assigned_target = self.assigned_targets[my_id]
                if self.calculate_distance_3d(my_plane, assigned_target) >= 40000:  # TODO：这个切换比较tricky
                    self.phase = 1  # 一开始是冲向热区(伪)
                else:
                    self.use_fake_heat_zone[my_id] = False
                    self.phase = 2  # 然后开启狗斗模式
            else:
                self.phase = 2
                
            for entity_info in obs.rws_infos:
                if entity_info.ind in self.full_enemy_plane_id_list or entity_info.ind in self.expired_missiles:
                    continue

                if my_id in entity_info.alarm_ind_list:
                    distance = self.calculate_distance_3d(my_plane, entity_info)
                    if entity_info.ind not in self.missile_tracks:
                        self.missile_tracks[entity_info.ind] = []
                        self.missile_plane_distance_tracks[entity_info.ind] = []
                    self.missile_tracks[entity_info.ind].append([entity_info.x, entity_info.y, entity_info.z])
                    self.missile_plane_distance_tracks[entity_info.ind].append(distance)

                    if entity_info.ind in self.dangerous_missiles and len(self.missile_plane_distance_tracks[entity_info.ind]) > 40:
                        if self.missile_plane_distance_tracks[entity_info.ind][-1] - self.missile_plane_distance_tracks[entity_info.ind][0] > 1000:
                            self.expired_missiles.add(entity_info.ind)
                            if debug_flag:
                                print("expired_missiles: ",self.expired_missiles)
                            continue

                    # 判断威胁
                    if (closest_missile is None or distance < self.calculate_distance_3d(my_plane, closest_missile)) and \
                            (self.calculate_distance_2d(my_plane, entity_info) < 30000):
                        self.dangerous_missiles.add(entity_info.ind)
                        if len(self.missile_tracks[entity_info.ind])>10:
                            _, _, _, is_ahead_of_enemy = is_facing_target(np.array(self.missile_tracks[entity_info.ind][-10:]), 
                                                        np.array(self.myplane_tracks[my_plane.ind][-10:]))
                            if is_ahead_of_enemy: # TODO: 只躲容易躲的弹，否则进攻就是最好的防守
                                self.phase = 3
                                closest_missile = entity_info

            if closest_missile:
                _, cos_theta, can_face_target_position, _ = is_facing_target(np.array(self.missile_tracks[closest_missile.ind][-10:]), 
                                                        np.array(self.myplane_tracks[my_plane.ind][-10:]),debug=debug_flag) 
                
                target_pos = Vector3(can_face_target_position[0],can_face_target_position[1],my_plane.z)
                action = self.get_action_cmd(target_pos, my_plane, "missile", debug = debug_flag)
                action[0] = 1  # 不改变高度可以让转向加快

                if cos_theta < 0.7:
                    if debug_flag:
                        print("比较好躲!!!")
                    raw_cmd_dict[my_id]['control'] = fly_with_alt_yaw_vel(my_plane, action, self.id_pidctl_dict[my_id])
                else:
                    if debug_flag:
                        print("不太好躲!!!交给专家!!!")
                
            if self.phase == 1:
                target_pos = self.assigned_targets[my_id]
                action = self.get_action_cmd(target_pos, my_plane, "fix_point", debug = debug_flag)
                raw_cmd_dict[my_id]['control'] = fly_with_alt_yaw_vel(my_plane, action, self.id_pidctl_dict[my_id])
            
            if my_plane.z > 6500:
                if debug_flag:
                    print("高度过低，紧急抬升！！!")
                    raw_cmd_dict[my_id]['control'][1] = -1
                    raw_cmd_dict[my_id]['control'][3] = 0.1

            # 以上是机动部分
            # ---------------------------------------------------------------------------------------------------
            # 以下是发弹部分
            # 关于发弹的优化不要硬绑在机动的状态里面，不然容易克制不住奇兵
            new_weapon_launch_info = {}
            if 'weapon' in raw_cmd_dict[my_id]:
                new_weapon_launch_info = self.get_weapon_launch_info(obs,my_plane, debug=debug_flag)
                if len(new_weapon_launch_info):
                    raw_cmd_dict[my_id]['weapon'] = new_weapon_launch_info
                else:
                    del raw_cmd_dict[my_id]['weapon']
            
            if debug_flag:
                tmp_mid_lock_list = [] if my_plane.is_uav else my_plane.mid_lock_list
                print("Step: ",self.run_counts,", ID: ", my_id, 
                    ", 位置：", [my_plane.x,my_plane.y,my_plane.z],
                    ", 速度 [v_north, v_east, v_down]: ", [my_plane.v_north,my_plane.v_east,my_plane.v_down],
                    ", 机动控制: ", raw_cmd_dict[my_id]["control"],
                    ", (中距)/近距弹锁定信息: ", tmp_mid_lock_list,my_plane.short_lock_list,
                    ", 发弹控制: ", new_weapon_launch_info)

        self.run_counts += 1
        return raw_cmd_dict


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
