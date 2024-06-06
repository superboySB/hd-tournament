import numpy as np
from sturnus.geo import *

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
        self.assign = {}
        self.missile_cds = {}
        self.missile_time = {}

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
        rws_infos = obs.rws_infos
        
        # 对每架我方飞机计算行动
        for ally_ind, ally in allies.items():
            # 按格式设置每架飞机的动作、武器指令
            weapon_launch_info = {}
            action = [0, 0, 0, 0]
            # if np.random.rand() < 0.5:
            #     action = [-0.5 + 1 * self.rng.random(), -0.06 + 0.1*self.rng.random(), 0.01 * np.random.rand(), 0.5 + 0.5 * np.random.rand()]
            if obs.sim_time > 1 and len(ally.mid_lock_list)>0 and ally.loadout.get('mid_missile', 0):
                        # input()
                weapon_launch_info = {
                    'type': 'mid_missile',
                    'target': ally.mid_lock_list[0],
                }
            cmd_dict[ally_ind] = {
                'control': action
            }
            if len(weapon_launch_info):
                cmd_dict[ally_ind]['weapon'] = weapon_launch_info
            
        # 返回指令。
        return cmd_dict