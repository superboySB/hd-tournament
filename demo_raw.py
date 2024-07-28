import json
import numpy as np
import pandas as pd
from hddf2sim.hddf2sim import HDDF2Sim
from hddf2sim.conf import default_conf
from agents.team_blue.blue_agent_demo import Agent as BlueAgent
from agents.team_blue.blue_agent_demo import Agent as RedAgent

# 读取场景配置文件
with open("scen.json", "r") as fin:
    scen = json.load(fin)

# 初始化模拟器
sim = HDDF2Sim(scen, use_tacview=True, save_replay=True, replay_path="replay.acmi")
sim.reset()

# 初始化红蓝双方的代理
red_agent = RedAgent('red')
blue_agent = BlueAgent('blue')

# 初始化控制组合计数器
control_counter = {}
first_appearance = {}
last_appearance = {}

def round_control(control):
    return tuple(round(c, 2) for c in control)

# 模拟对局
step_count = 0
while not sim.done:
    step_count += 1
    
    # 获取红方观察并执行一步
    red_obs = sim.get_obs(side='red')
    red_cmd_dict = red_agent.step(red_obs)
    sim.send_commands(red_cmd_dict, cmd_side='red')
    
    # 获取蓝方观察并执行一步
    blue_obs = sim.get_obs(side='blue')
    blue_cmd_dict = blue_agent.step(blue_obs)
    sim.send_commands(blue_cmd_dict, cmd_side='blue')
    
    # 记录红方的控制组合
    for agent_id, cmd in red_cmd_dict.items():
        control = round_control(tuple(cmd['control']))  # 将列表转换为元组，并四舍五入小数点后两位
        if control in control_counter:
            control_counter[control] += 1
            last_appearance[control] = step_count
        else:
            control_counter[control] = 1
            first_appearance[control] = step_count
            last_appearance[control] = step_count
    
    # 记录蓝方的控制组合
    for agent_id, cmd in blue_cmd_dict.items():
        control = round_control(tuple(cmd['control']))  # 将列表转换为元组，并四舍五入小数点后两位
        if control in control_counter:
            control_counter[control] += 1
            last_appearance[control] = step_count
        else:
            control_counter[control] = 1
            first_appearance[control] = step_count
            last_appearance[control] = step_count
    
    # 进行下一步模拟
    sim.step()

# 将控制组合及其出现次数保存为CSV文件
control_data = [
    {
        'control': control,
        'count': count,
        'first_appearance': first_appearance[control],
        'last_appearance': last_appearance[control]
    }
    for control, count in control_counter.items()
]
df = pd.DataFrame(control_data)
df = df.sort_values(by='first_appearance', ascending=True)
df.to_csv("count_command.csv", index=False)

print("控制组合及其出现次数和出现步数已保存至 count_command.csv")
