import numpy as np
import math
import agents.test_red.pid as pid
import agents.test_red.utils as utils
from sturnus.geo import *
from agents.test_red.pid import FlyPid
# 选手定义名为Agent的类， 
# 1. 实现 初始化函数(__init__)， 
# 2. 实现 每一帧读取处理态势并返回指令的函数（step)。
target_info = 0
class Agent:
    def __init__(self, side) -> None:
        # 初始化函数
        # 初始化一些相关变量，工具类。
        self.side = side
        self.cmd_id = 0
        self.run_counts = 0
        self.tar_id = 0
        self.ini_pid = False
        self.id_pidctl_dict = {}

    def weapon_cmd(self, type:str, target:int) -> dict:
        return {"type": type, "target": target}
    
    def get_action_cmd(self, target, plane):
        """_summary_

        Args:
            target (_type_): _description_

        Returns:
            list: [delta_alt, delta_yaw(角度), delta_vel]
        """
        
        plane_pos = utils.Vector3(
        plane.x, plane.y, plane.z
        )
        target_pos = utils.Vector3(
            target.x, target.y, target.z
        )
        action = np.zeros(3, dtype=int)
        if target.height - plane.height > 500:# 升高
            action[0] = 0
            print("升高", end=' ')
        elif target.height - plane.height < -500:# 降高
            action[0] = 2
            print("降高", end=' ')
        else:
            action[0] = 1# 保持
            print("保持", end=' ')
        
        to_target_vec = target_pos - plane_pos
        target_yaw = np.arctan2(to_target_vec.y, to_target_vec.x)
        delta_yaw = utils.degrees_limit(math.degrees(target_yaw - plane.yaw))
        
        if delta_yaw > 30:# 右转 30度
            action[1] = 6
            print("右转 30度", end=' ')
        elif 15 < delta_yaw:# 右转 15度
            action[1] = 5
            print("右转 15度", end=' ')
        elif 5 < delta_yaw:# 右转 5度
            action[1] = 4
            print("右转 5度", end=' ')
        elif -5 < delta_yaw:# 左转 5度
            action[1] = 3
            print("直走", end=' ')
        elif -15 < delta_yaw:# 直走
            action[1] = 2
            print("左转 5度", end=' ')
        elif -30 < delta_yaw:# 左转 15度
            action[1] = 1
            print("左转 15度", end=' ')
        else:
            action[1] = 0# 左转30度
            print("左转 30度", end=' ')
            
        if target.tas - plane.tas > 10:# 加速
            action[2] = 0
            print("加速", end=' ')
        elif target.tas - plane.tas < -10:# 尽量匀速
            action[2] = 2
            print("减速", end=' ')
        else:
            action[2] = 1# 减速
            print("匀速", end=' ')
            
        return action
    
    def step(self, obs, blue_obs, data_queue=None):
        cmd_dict = {}
        global target_point, target_info
        self.run_counts += 1
        # 获取此步长的地方信息
        for id, tar_plane in blue_obs.my_planes.items():
            if self.tar_id == 0:
                self.tar_id = tar_plane.ind
                print("目标：", self.tar_id)
                target_info = tar_plane
                break
            elif self.tar_id == id:
                target_info = tar_plane
                break
        
        if self.tar_id:
            for id, plane in obs.my_planes.items():
                if not self.ini_pid:
                    self.id_pidctl_dict[id] = FlyPid()
                cmd = {}
                if self.cmd_id == 0 and id != self.tar_id:
                    print("控制：", id)
                    self.cmd_id = id
                if id == self.cmd_id:
                    action = self.get_action_cmd(target_info, plane)
                    cmd['control'] = utils.fly_with_alt_yaw_vel(plane, obs.sim_time, action, fly_pid=self.id_pidctl_dict[id], data_queue=data_queue)
                    print(f"{self.cmd_id} to {self.tar_id} action: {action}, cmd: {cmd['control']}, omega_r: {math.degrees(plane.omega_r)}")
                    
                    cmd_dict[id] = cmd
            self.ini_pid = True
        return cmd_dict