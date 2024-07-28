import json
import numpy as np
from hddf2sim.hddf2sim import HDDF2Sim
from hddf2sim.conf import default_conf

from agents.team_blue.blue_agent_demo import Agent as BlueAgent

# from agents.team_blue.blue_agent_demo import Agent as RedAgent
from agents.houlang.agent import Agent as RedAgent

with open("scen_0721.json", "r") as fin:
    scen = json.load(fin)

sim = HDDF2Sim(scen, use_tacview=True, save_replay=True, replay_path="replay.acmi")
sim.reset()
red_agent = RedAgent('red')
blue_agent = BlueAgent('blue')
num_step = 0

while not sim.done:
    num_step += 1 
    cmds = []
    red_obs = sim.get_obs(side='red')
    red_cmd_dict = red_agent.step(red_obs)

    # for key, value in red_obs.my_planes.items():
    #     if num_step % 10 == 0:
    #         print(f"Step: {num_step}")
    #         print(f"Plane ID: {key}, My plane coordinates: (x: {value.x}, y: {value.y}, z: {value.z}, yaw: {value.yaw}, v_north: {value.v_north}, v_east: {value.v_east}, v_down: {value.v_down})")
    #         print(f"Control: aileron={red_cmd_dict[key]['control'][0]}, elevator={red_cmd_dict[key]['control'][1]}, rudder={red_cmd_dict[key]['control'][2]}, throttle={red_cmd_dict[key]['control'][3]}")

    sim.send_commands(red_cmd_dict, cmd_side='red')
    blue_obs = sim.get_obs(side='blue')
    blue_cmd_dict = blue_agent.step(blue_obs)
    cmds.extend(blue_cmd_dict)
    sim.send_commands(blue_cmd_dict, cmd_side='blue')
    sim.step()
input("单局推演结束，按Enter退出。")