import numpy as np
from sturnus.geo import *
from .blue_agent_demo import Agent as BaseAgent

# 选手定义名为Agent的类， 
# 1. 实现 初始化函数(__init__)， 
# 2. 实现 每一帧读取处理态势并返回指令的函数（step)。
class Agent(BaseAgent):
    def __init__(self, side) -> None:
        # 初始化函数
        # 初始化一些相关变量，工具类。
        super().__init__(side)

    def step(self, obs):
        raw_cmd_dict =  super().step(obs)
        # print(raw_cmd_dict)

        for key, value in raw_cmd_dict.items():
            if 'weapon' in value:
                del value['weapon']

        return raw_cmd_dict