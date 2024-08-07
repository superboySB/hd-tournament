import json
import numpy as np
from hddf2sim.hddf2sim import HDDF2Sim
from hddf2sim.conf import default_conf

# from agents.team_blue.blue_agent_demo import Agent as BlueAgent
# from agents.houlang.agent import Agent as BlueAgent
from agents.houlang_dev.agent import Agent as BlueAgent

from agents.team_blue.blue_agent_demo import Agent as RedAgent
# from agents.houlang_dev.agent import Agent as RedAgent

with open("scen_fadan.json", "r") as fin:
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

    sim.send_commands(red_cmd_dict, cmd_side='red')
    blue_obs = sim.get_obs(side='blue')
    blue_cmd_dict = blue_agent.step(blue_obs)
    cmds.extend(blue_cmd_dict)

    # if num_step % 10 == 0:
    #     print(f"\nStep {num_step}")
    #     for key, value in blue_obs.my_planes.items():
    #         my_plane_info = value
    #         print(f"[State] x: {my_plane_info.x}, y: {my_plane_info.y}, z: {my_plane_info.z}, \
    #             roll: {my_plane_info.roll}, pitch: {my_plane_info.pitch}, yaw: {my_plane_info.yaw}, \
    #             v_down: {my_plane_info.v_down}, v_east: {my_plane_info.v_east}, v_north: {my_plane_info.v_north}")
    #     for key, value in blue_cmd_dict.items():
    #         my_plane_action = value
    #         print(f"[aileron, elevator, rudder, throttle] {my_plane_action['control']}")
    
    sim.send_commands(blue_cmd_dict, cmd_side='blue')
    sim.step()
input("单局推演结束，按Enter退出。")