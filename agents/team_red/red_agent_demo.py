import numpy as np
from sturnus.geo import *

def do_rotate(plane, oblique='right', target_height=0, sp = 1.0):

    """
        转弯
    """

    turn_target_roll = (1.35 if oblique=='right' else -1.35)
    target_sign = 1 if oblique=='right' else -1
    turn_over_roll = turn_target_roll * 1.28
    
    # 根据当前plane的roll确定下一步的aileron
    if stdr(plane.roll)*target_sign < turn_target_roll*target_sign:
        aileron = 0.3 * target_sign
    elif stdr(plane.roll)*target_sign > turn_over_roll*target_sign:
        aileron = -0.5 * target_sign
    else:
        aileron = -0.3 * target_sign

    # 计算target_pitch
    cur_plane_pos_z = -plane.pos.z if plane.pos.z < 0 else plane.pos.z
    if np.abs(plane.pos.z - target_height)<5e2:
        target_pitch = 0.05
    elif cur_plane_pos_z > target_height:
        target_pitch = -0.1
    else:
        target_pitch = 0.2

    # 根据当前飞机的pitch和target_pitch 确定下一步的elevator
    if plane.pitch - target_pitch > 0.3:
        elevator = 0.5
    elif plane.pitch - target_pitch > 0.1:
        elevator = 0.1
    else:
        elevator = -0.7

    action = [
        aileron, elevator, 0, 
        sp
    ]
    return action

def do_aiming(plane, target):

    to_target_vec = (target.pos - plane.pos).unit_vec()
    to_target_yaw, to_target_pitch = vec_to_yp(to_target_vec)   # 根据vec得到target_yaw和target_pitch
    delta_yaw = get_turn_delta(plane.yaw, to_target_yaw)   # 得到需要旋转的最小delta_yaw
    delta_pitch = get_turn_delta(plane.pitch, to_target_pitch)   # 得到需要旋转的最小delta_pitch
    elevator = 0.1 * (-np.sign(delta_pitch))
    rudder = 0.3 * (-np.sign(delta_yaw))
    aileron = 0.2 if plane.roll < 0 else -0.1
    throttle = 1.0
    act =  [aileron, elevator, rudder, throttle]

    return act

def do_pointing(plane, target_yaw, target_height):
    turn_delta = get_turn_delta(plane.yaw, target_yaw)

    # 确定转向
    if turn_delta > 0:
        oblique = 'right'
    else:
        oblique = 'left'

    action = do_rotate(plane, oblique=oblique, target_height = target_height)
    return action

def attack_move(plane, target):
    plane_pos = Vec3(
        plane.x, plane.y, plane.z
    )
    target_pos = Vec3(
        target.x, target.y, target.z
    )
    to_target_vec = target_pos - plane_pos
    target_yaw = np.arctan2(to_target_vec.y, to_target_vec.x)
    return do_pointing(plane, target_yaw, target_pos.z)

def fly_down(plane):
    """
        向下飞
    """
    if plane.roll>0:
        aileron = 1.0
    else:
        aileron = -1.0
    if np.abs(plane.roll) > np.pi/2.0 and plane.pitch>-1.0:
        elevator = -1.0
    else:
        elevator = 0.0
    action = [aileron, elevator, 0, 1.0]
    return action

def pull_up(plane):
    """
        向上拉起
    """
    if np.abs(plane.roll)<0.2:
        aileron = 0.0
    else:
        aileron = np.clip(-0.8 * plane.roll , -1,1)
    
    if np.abs(plane.roll) < 1.3:
        elevator = -1.0
    else:
        elevator = 0.0
    if plane.cas < 200.0:
        thr = 1.0
    else:
        thr = 0.0
    action = [aileron, elevator, 0, thr]
    return action

def back_to_region(ally):
    """
        太靠近边界了，往回飞
    """

    if np.abs(ally.pos.x)>10e3 and np.abs(ally.pos.y)>45e3:
        target_yaw = np.arctan2(-ally.pos.y, -ally.pos.x)
    elif np.abs(ally.pos.x)>10e3:
        target_yaw = np.arctan2(0, -ally.pos.x)
    else:
        target_yaw = np.arctan2(-ally.pos.y, 0)
   
    action = do_pointing(ally, target_yaw, ally.z)
    if ally.cas > 200:
        thr = 0.0
    else:
        thr = 1.0
    action[3] = thr
    # print(ally.pos, target_yaw)
    return action

# 选手定义名为Agent的类， 
# 1. 实现 初始化函数(__init__)， 
# 2. 实现 每一帧读取处理态势并返回指令的函数（step)。
class Agent:
    def __init__(self, side) -> None:
        # 初始化函数
        # 初始化一些相关变量，工具类。
        self.side = side
        self.rng = np.random.default_rng()

        # 初始化一些策略状态， 统计变量。
        self.missile_cds = {}
        self.mid_missile_time = {}    # 记录中距弹上一次打弹时间，用于设置打弹cd
        self.short_missile_time = {}    # 记录近距弹上一次打弹时间，用于设置打弹cd
        self.mid_lock_time = 0
        self.mid_lock_target_ind = 0

    def step(self, obs):
        # 每一帧读取处理态势并返回指令的函数
        # 按格式创建命令字典。
        cmd_dict = {}

        # 预处理态势
        allies = obs.my_planes
        for ally in allies.values():
            ally.pos = Vec3(
                ally.x, ally.y, ally.z
            )

        enemies = obs.enemy_planes
        e_inds = []
        for enemy in enemies.values():
            e_inds.append(enemy.ind)
            enemy.pos = Vec3(
                enemy.x, enemy.y, enemy.z
            )
        awacs_infos = obs.awacs_infos

        self.mid_lock_time = 0
        
        # 对每架我方飞机计算行动
        for ally_ind, ally in allies.items():
            weapon_launch_info = {}

            if len(ally.mid_lock_list)>0 and \
                ally.loadout.get('mid_missile', 0)>0 and obs.sim_time - self.mid_missile_time.get(ally_ind, 0) > 10.0:
                    weapon_launch_info = {
                        'type': 'mid_missile',
                        'target': ally.mid_lock_list[0],
                    }
                    self.mid_missile_time[ally_ind] = obs.sim_time
                    self.mid_lock_time = obs.sim_time
                    self.mid_lock_target_ind = ally.mid_lock_list[0]

            if len(ally.short_lock_list)>0 and \
                ally.loadout.get('short_missile', 0)>0 and obs.sim_time - self.short_missile_time.get(ally_ind, 0) > 5.0:
                    weapon_launch_info = {
                        'type': 'short_missile',
                        'target': ally.short_lock_list[0],
                    }
                    self.short_missile_time[ally_ind] = obs.sim_time
            
            if ally.pitch > 1.0 or ally.pos.z<-3e3:
                action = fly_down(ally)
            elif (ally.pos.z > 8e3) or  (ally.pos.z > 6e3 and ally.pitch< -0.7):
                action = pull_up(ally)
            elif np.abs(ally.pos.x) > 10e3 or np.abs(ally.pos.y) > 45e3:
                action = back_to_region(ally)
            elif obs.sim_time - self.mid_lock_time < 30.0 and self.mid_lock_target_ind in enemies:
                target = enemies[self.mid_lock_target_ind]
                action = do_aiming(ally, target)
            else:
                # 整合雷达和预警信息。 并针对任一敌机做攻击机动。
                if len(enemies) or len(awacs_infos):
                    if len(enemies):
                        enemy = list(enemies.values())[0]
                    else:
                        enemy = awacs_infos[0]
                    action = attack_move(ally,  enemy)

                else:
                    action = [0, 0, 0, 0.8]
            
            # 按格式设置每架飞机的动作、武器指令

            cmd_dict[ally_ind] = {
                'control': action
            }
            if len(weapon_launch_info):
                cmd_dict[ally_ind]['weapon'] = weapon_launch_info
        # 返回指令。

        return cmd_dict