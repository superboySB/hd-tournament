import numpy as np
import math
from sturnus.geo import *
from .funcs import FlyPid, Vector3, degrees_limit, is_facing_missile  # 确保 FlyPid 模块正确引用

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
        self.run_counts = 0
        self.phase = 1  # 1 for approach, 2 for circling
        self.heat_zone_center = Vector3(0, 0, -3000)
        self.heat_zone_radius = 15000
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
        self.mid_lock_target_ind = 0
    
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
            action[0] = 1# 保持
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
        if mode == "fix_point":
            if plane.is_uav:
                action[2] = 0
                if debug:
                    print("加速", end=' ')
            else:
                action[2] = 1
                if debug:
                    print("匀速", end=' ')
        elif mode == "plane":
            if target.tas - plane.tas > 10:
                action[2] = 0
                if debug:
                    print("加速", end=' ')
            elif target.tas - plane.tas < -10:
                action[2] = 2
                if debug:
                    print("减速", end=' ')
            else:
                action[2] = 1
                if debug:
                    print("匀速", end=' ')
        elif mode == "missile":
            action[2] = 0
            if debug:
                print("加速", end=' ')
        else:
            raise NotImplementedError
            
        return action

    def get_weapon_launch_info(self, obs, my_plane):
        weapon_launch_info = {}
        if len(my_plane.mid_lock_list)>0 and \
            my_plane.loadout.get('mid_missile', 0)>0 and obs.sim_time - self.mid_missile_time.get(my_plane.ind, 0) > 10.0:
                weapon_launch_info = {
                    'type': 'mid_missile',
                    'target': my_plane.mid_lock_list[0],
                }
                self.mid_missile_time[my_plane.ind] = obs.sim_time
                self.mid_lock_target_ind = my_plane.mid_lock_list[0]

        if len(my_plane.short_lock_list)>0 and \
            my_plane.loadout.get('short_missile', 0)>0 and obs.sim_time - self.short_missile_time.get(my_plane.ind, 0) > 5.0:
                weapon_launch_info = {
                    'type': 'short_missile',
                    'target': my_plane.short_lock_list[0],
                }
                self.short_missile_time[my_plane.ind] = obs.sim_time

        return weapon_launch_info

    def step(self, obs):        
        debug_flag = False
        if self.run_counts % 10 == 0 :
            print("\n------------------------------------------------------------------\n")
            debug_flag = True
        if len(obs.awacs_infos) and len(self.full_enemy_plane_id_list)==0:
            for awacs_i in obs.awacs_infos:
                self.full_enemy_plane_id_list.append(awacs_i.ind)
                self.enemy_plane_tracks[awacs_i.ind] = []
        
        if len(obs.enemy_planes):  # TODO: 在这里添加敌机的跟踪逻辑
            pass

        cmd_dict = {}
        for my_id, my_plane in obs.my_planes.items():
            if not self.ini_pid or my_id not in self.id_pidctl_dict:
                self.id_pidctl_dict[my_id] = FlyPid()

            if my_id not in self.myplane_tracks:
                self.myplane_tracks[my_id] = []
            self.myplane_tracks[my_id].append([my_plane.x, my_plane.y, my_plane.z])

            closest_missile = None
            cmd = None
            if self.calculate_distance_2d(my_plane,self.heat_zone_center) >= 15000:
                self.phase = 1  # 一开始是冲向热区
            else:
                self.phase = 2 # TODO: 然后开启狗斗模式

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
                    if closest_missile is None or distance < self.calculate_distance_3d(my_plane, closest_missile):
                        if len(self.missile_tracks[entity_info.ind])>10:
                            can_face_missile, _, is_ahead_of_enemy = is_facing_missile(np.array(self.missile_tracks[entity_info.ind][-10:]), 
                                                        np.array(self.myplane_tracks[my_plane.ind][-10:]))
                            if can_face_missile and is_ahead_of_enemy: # TODO： 只躲能躲的弹，否则进攻就是最好的防守
                                self.phase = 3
                                self.dangerous_missiles.add(entity_info.ind)
                                closest_missile = entity_info

            if closest_missile:
                _, can_face_target_position, _ = is_facing_missile(np.array(self.missile_tracks[closest_missile.ind][-10:]), 
                                                        np.array(self.myplane_tracks[my_plane.ind][-10:]), debug = debug_flag)                                               
                target_pos = Vector3(can_face_target_position[0],can_face_target_position[1],can_face_target_position[2])
                action = self.get_action_cmd(target_pos, my_plane, "missile", debug = debug_flag)
                cmd = {'control': fly_with_alt_yaw_vel(my_plane, action, self.id_pidctl_dict[my_id])}
                
            if self.phase == 1:
                target_pos = self.heat_zone_center
                action = self.get_action_cmd(target_pos, my_plane, "fix_point", debug = debug_flag)
                cmd = {'control': fly_with_alt_yaw_vel(my_plane, action, self.id_pidctl_dict[my_id])}

            if self.phase == 2:
                target_pos = self.heat_zone_center
                action = self.get_action_cmd(target_pos, my_plane, "fix_point", debug = debug_flag)
                cmd = {'control': fly_with_alt_yaw_vel(my_plane, action, self.id_pidctl_dict[my_id])}
                
            # 以上是机动部分
            # ---------------------------------------------------------------------------------------------------
            # 以下是发弹部分
            #  TODO: 关于发弹的优化还没做，这个可以单独做，不要硬绑在机动的状态里面，不然容易克制不住奇兵
            weapon_launch_info = self.get_weapon_launch_info(obs,my_plane)
            if len(weapon_launch_info):
                cmd['weapon'] = weapon_launch_info
            
            # 拼接命令，准备机动+发弹控制
            cmd_dict[my_id] = cmd
            if debug_flag:
                tmp_mid_lock_list = [] if my_plane.is_uav else my_plane.mid_lock_list
                print("Step: ",self.run_counts,", ID: ", my_id, 
                    ", 位置：", [my_plane.x,my_plane.y,my_plane.z],
                    ", 速度：", [my_plane.v_north,my_plane.v_east,my_plane.v_down],
                    ", 机动控制: ", cmd["control"],
                    ", (中距)/近距弹锁定信息: ", tmp_mid_lock_list,my_plane.short_lock_list,
                    ", 发弹控制: ", weapon_launch_info)

        self.run_counts += 1
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
