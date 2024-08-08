import math
import numpy as np

'''
李超PID相关
'''
# Kp、Ki、Kd：分别是比例、积分和微分增益参数，决定了 PID 控制器对误差的响应速度和稳定性。
# setpoint：控制目标值，系统希望达到的设定值。
# output_limits：限制输出范围的元组，用于避免控制输出过大或过小。
# windup_guard：积分风暴保护，用于限制积分累加值，防止积分项过大导致控制器不稳定。
# previous_error：上一次计算的误差值，用于计算微分项。
# integral：误差的积分累加值，用于计算积分项。
last_target_pitch = 0
last_target_heading = 0
last_target_roll = 0
norm_delta_altitude = np.array([500, 0, -500])
norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, -np.pi / 36, 0, np.pi / 36, np.pi / 12, np.pi / 6])
norm_delta_velocity = np.array([0.05, 0, -0.05])

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(None, None), windup_guard=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.windup_guard = windup_guard
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, measured_value, length=1):
        error = self.setpoint - measured_value
        error /= length
        self.integral += error

        # Apply windup guard
        if self.windup_guard is not None:
            self.integral = max(min(self.integral, self.windup_guard), -self.windup_guard)

        derivative = error - self.previous_error
        self.previous_error = error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(output, self.output_limits[0])
        if self.output_limits[1] is not None:
            output = min(output, self.output_limits[1])

        return output
    
    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0

# 增大 𝐾𝑝: 可以提高系统响应速度，但可能增加过冲和振荡。
# 减小 𝐾𝑝: 可以减少过冲和振荡，但会降低系统响应速度。
# 增大 Ki: 可以消除稳态误差，但可能增加过冲和振荡。
# 减小 𝐾𝑖: 可以减少过冲和振荡，但可能导致稳态误差存在。 
# 增大 Kd: 可以减少振荡和过冲，但可能使系统响应变慢。
# 减小 Kd: 可以加快系统响应，但可能增加振荡和过冲。
class FlyPid:
    def __init__(self):
        # 初始化 PID 控制器
        self.pid_aileron = PID(Kp=0.8, Ki=0.01, Kd=0.1, output_limits=(-1, 1), windup_guard=20.0)
        self.pid_elevator = PID(Kp=0.3, Ki=0.02, Kd=0.2, output_limits=(-1, 1), windup_guard=10.0)
        self.pid_rudder = PID(Kp=0.4, Ki=0.01, Kd=0.1, output_limits=(-1, 1), windup_guard=1.0)
        self.pid_throttle = PID(Kp=0.4, Ki=0.01, Kd=0.1, output_limits=(0, 1), windup_guard=1.0)

    def set_tar_value(self, tar_pitch_rate, tar_yaw_rate, tar_roll_rate):
        self.pid_elevator.set_setpoint(math.degrees(tar_pitch_rate))
        self.pid_rudder.set_setpoint(math.degrees(tar_yaw_rate))
        self.pid_aileron.set_setpoint(math.degrees(tar_roll_rate))
        
    def get_control_cmd(self, omega_p, omega_q, omega_r):
        current_roll = math.degrees(omega_p)
        current_pitch = math.degrees(omega_q)
        current_yaw = math.degrees(omega_r)
        
        # 计算控制输入
        control_aileron = self.pid_aileron.compute(current_roll, 90)
        control_elevator = -self.pid_elevator.compute(current_pitch)
        # control_rudder = -self.pid_rudder.compute(current_yaw, 30)
        # control_throttle = pid_throttle.compute( current_airspeed, 1)

        return [control_aileron, control_elevator, 0, 1]

def fly_with_alt_yaw_vel(plane, action:list, fly_pid):
    """_summary_

    Args:
        action (list): [(0-2), (0-4), (0-2)] ,

    Returns:
        list: [aileron, elevator, rudder, throttle]
    """
    global last_target_pitch, last_target_heading, last_target_roll
    # 确定转向
    temp_turn = math.degrees(norm_delta_heading[action[1]])
    # 根据当前状态想移动的角度来计算目标滚转角度
    rate = (abs(temp_turn) / 35)
    # if rate >= 1:
    #     rate = 0.99
    if abs(temp_turn) < 4:
        target_roll = 0
    elif temp_turn > 0: 
        target_roll = math.radians(90) * rate # 右转
    else:
        target_roll = math.radians(-90) * rate # 左转
    
    target_pitch = np.arctan2(norm_delta_altitude[action[0]], 500)
    
    # 设置目标姿态角角速度
    fly_pid.set_tar_value(target_pitch - plane.pitch, norm_delta_heading[action[1]], target_roll - plane.roll)
    cmd_list = fly_pid.get_control_cmd(plane.omega_p, plane.omega_q, plane.omega_r)
    
    # 控制加力来改变速度
    if norm_delta_velocity[action[2]] < 0:
        cmd_list[3] = 0
    elif norm_delta_velocity[action[2]] == 0:
        cmd_list[3] = 0.395 # 匀速
        
    # 限制控制俯仰轴的指令值，防止飞机失控
    if abs(cmd_list[1])  > 0.7:
        cmd_list[1] = 0.7 * np.sign(cmd_list[1])
        
    # 飞机倒过来飞时特殊处理，俯仰轴取反方向
    if -90 >= math.degrees(plane.roll) or math.degrees(plane.roll) >= 90:
        cmd_list[1] = -1 * cmd_list[1]
    
    last_target_pitch = target_pitch
    last_target_heading = plane.yaw + math.radians(temp_turn)
    last_target_roll = target_roll
    
    return cmd_list

