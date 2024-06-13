import numpy as np
from sturnus.geo import *

def do_rotate(plane, oblique='right', target_height=0, sp = 1.0):

    """
        转弯
    """

    turn_target_roll = (1.35 if oblique=='right' else -1.35)
    target_sign = 1 if oblique=='right' else -1
    turn_over_roll = turn_target_roll * 1.28  # 定义超转横滚角度，是目标横滚角度的1.28倍，用于确定何时需要减小横滚角度以防过度转弯。
    
    # 根据当前plane的roll确定下一步的aileron
    # 如果调整后的飞机当前横滚角度（调整为相对于转向的正负值）小于目标横滚角度，则增加副翼偏转量以增大横滚。
    if stdr(plane.roll)*target_sign < turn_target_roll*target_sign:
        aileron = 0.3 * target_sign
    # 如果飞机的横滚角度大于超转横滚角度，减小副翼偏转量以减少横滚。
    elif stdr(plane.roll)*target_sign > turn_over_roll*target_sign:
        aileron = -0.5 * target_sign
    # 如果飞机横滚角度在目标和超转之间，略微减小副翼偏转量以微调横滚。
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
    # 如果当前俯仰与目标俯仰差值大于0.3，增加升降舵偏转量以增大俯仰角。
    if plane.pitch - target_pitch > 0.3:
        elevator = 0.5
    # 如果差值大于0.1，轻微调整升降舵以微调俯仰。
    elif plane.pitch - target_pitch > 0.1:
        elevator = 0.1
    # 如果当前俯仰小于或等于目标俯仰，大幅下调升降舵以减少俯仰角。
    else:
        elevator = -0.7

    # 组装动作列表，包含副翼、升降舵、方向舵（这里未调整，所以为0）和油门设置。
    action = [
        aileron, elevator, 0, sp
    ]
    return action

# 函数主要负责调整飞机的姿态以精确瞄准目标。其核心功能是计算飞机需要进行的最小角度调整，以确保飞机的瞄准线与目标位置对齐。
# 这里涉及的主要操作是偏航和俯仰的调整，使得飞机的武器（如导弹）能够正确锁定并攻击目标。
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
    # 用于后续计算飞机应该朝哪个方向移动以便对准目标。
    target_yaw = np.arctan2(to_target_vec.y, to_target_vec.x)
    # 负责调整飞机的偏航和俯仰，以确保飞机可以正确地朝向并接近目标的位置
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
        for enemy in enemies.values():
            enemy.pos = Vec3(
                enemy.x, enemy.y, enemy.z
            )
        awacs_infos = obs.awacs_infos

        self.mid_lock_time = 0
        
        # 对每架我方飞机计算行动
        for ally_ind, ally in allies.items():
            weapon_launch_info = {}
            
            # 这两段代码体现了在模拟战斗中如何根据武器系统状态、目标锁定情况和冷却时间等因素来决定何时发射导弹。
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
            
            # 前三个条件都是为了防止越界
            if ally.pitch > 1.0 or ally.pos.z<-3e3:
                action = fly_down(ally)
            elif (ally.pos.z > 8e3) or  (ally.pos.z > 6e3 and ally.pitch< -0.7):
                action = pull_up(ally)
            elif np.abs(ally.pos.x) > 10e3 or np.abs(ally.pos.y) > 45e3:
                action = back_to_region(ally)
            # 调整飞机的姿态以精确瞄准目标
            # obs.sim_time - self.mid_lock_time < 30.0：
            # 检查自上次成功锁定中距目标后是否过去了少于30秒。
            # 如果是，这表示中距目标锁定仍然是较新的，可能仍然在有效范围内，飞机可以继续执行对该目标的行动
            # self.mid_lock_target_ind in enemies：
            # 检查之前锁定的中距目标是否仍然存在于当前的敌机列表中。这是必要的，因为目标可能已袻击落、逃离战场或因其他原因不再被视为威胁。
            # 当这两个条件同时满足时，代码段将执行该条件下的行动，这通常意味着飞机将继续对之前锁定的敌机目标进行追踪或攻击。
            elif obs.sim_time - self.mid_lock_time < 30.0 and self.mid_lock_target_ind in enemies:
                target = enemies[self.mid_lock_target_ind]
                action = do_aiming(ally, target)
            # 整合雷达和预警信息。 并针对任一敌机做攻击机动。
            else:                
                if len(enemies) or len(awacs_infos):
                    if len(enemies):  # 优先用enemies，其次再用awacs
                        enemy = list(enemies.values())[0]
                    else:
                        enemy = awacs_infos[0]
                    action = attack_move(ally,  enemy)

                else: # 油门设置为0.8，表明飞机保持较高的推力维持当前飞行状态。
                    action = [0, 0, 0, 0.8]
            
            # 按格式设置每架飞机的动作、武器指令
            cmd_dict[ally_ind] = {
                'control': action
            }
            if len(weapon_launch_info):
                cmd_dict[ally_ind]['weapon'] = weapon_launch_info
        # 返回指令。
        return cmd_dict