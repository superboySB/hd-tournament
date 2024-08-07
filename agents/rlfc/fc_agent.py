import numpy as np
from sturnus.geo import *
from .funcs import create_fc_model,set_target,get_obs,get_control_action,action2cmd



# 选手定义名为Agent的类， 
# 1. 实现 初始化函数(__init__)， 
# 2. 实现 每一帧读取处理态势并返回指令的函数（step)。
class Agent:
    def __init__(self, side) -> None:
        # 初始化函数

        self.path = 'fc_model\\discrete3'
        self.model = create_fc_model(self.path)
    def step(self, obs , target):
        # 每一帧读取处理态势并返回指令的函数

        # 按格式创建命令字典。
        cmd_dict = {}
        
        # 对每架我方飞机计算行动
        for pid,info in obs.my_planes.items():
          target = set_target(info,target)
          fc_obs = get_obs(info,target)
          action = self.model(fc_obs)
          action = get_control_action(action)
          cmd = action2cmd(action)
          cmd_dict[pid] = cmd
        
           
        # 返回指令。
        return cmd_dict