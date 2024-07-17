import json
import numpy as np
from hddf2sim.hddf2sim import HDDF2Sim
from hddf2sim.conf import default_conf

# from agents.team_blue.blue_agent_new import Agent as BlueAgent
from agents.team_blue.blue_agent_demo import Agent as BlueAgent
from agents.houlang.agent import Agent as RedAgent

with open("scen.json", "r") as fin:
    scen = json.load(fin)

sim = HDDF2Sim(scen, use_tacview=True, save_replay=True, replay_path="replay.acmi")
sim.reset()
red_agent = RedAgent('red')
blue_agent = BlueAgent('blue')

while not sim.done:
    cmds = []
    red_obs = sim.get_obs(side='red')
    red_cmd_dict = red_agent.step(red_obs)
    sim.send_commands(red_cmd_dict, cmd_side='red')
    blue_obs = sim.get_obs(side='blue')
    blue_cmd_dict = blue_agent.step(blue_obs)
    cmds.extend(blue_cmd_dict)
    sim.send_commands(blue_cmd_dict, cmd_side='blue')
    sim.step()
input("单局推演结束，按Enter退出。")