import math
import numpy as np
# Kp、Ki、Kd：分别是比例、积分和微分增益参数，决定了 PID 控制器对误差的响应速度和稳定性。
# setpoint：控制目标值，系统希望达到的设定值。
# output_limits：限制输出范围的元组，用于避免控制输出过大或过小。
# windup_guard：积分风暴保护，用于限制积分累加值，防止积分项过大导致控制器不稳定。
# previous_error：上一次计算的误差值，用于计算微分项。
# integral：误差的积分累加值，用于计算积分项。

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

# Reformat the print output for easier copying and pasting
def reformat_array_for_print(name, arr):
    print(f"{name}: [")
    for row in arr:
        print(f"    {list(row)},")
    print("]")

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance(self, point):
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2 + (self.z - point.z)**2)

def degrees_limit(angle):
    if angle > 180:
        return angle - 360
    elif angle < -180:
        return angle + 360
    return angle

def is_facing_target(target_posisions, aircraft_positions, debug=False):
    # 时间间隔
    delta_t = 0.05

    # 计算速度向量（方向）
    aircraft_directions = (aircraft_positions[1:] - aircraft_positions[:-1]) / delta_t
    target_directions = (target_posisions[1:] - target_posisions[:-1]) / delta_t

    # 归一化方向向量
    aircraft_directions /= np.linalg.norm(aircraft_directions, axis=1)[:, None]
    target_directions /= np.linalg.norm(target_directions, axis=1)[:, None]

    # 计算相对位置向量
    relative_positions = target_posisions[:-1] - aircraft_positions[:-1]

    # 计算导弹相对位置的单位方向向量
    relative_directions = relative_positions / np.linalg.norm(relative_positions, axis=1)[:, None]

    # 计算飞机和导弹的夹角余弦值
    cos_theta = np.sum(aircraft_directions * relative_directions, axis=1)
    facing_target = cos_theta > 0

    # 计算目标位置
    target_position = target_posisions[-1] - target_directions[-1] * 500

    # 判断飞机是否在导弹的前方
    # 获取最新的导弹位置和方向
    target_last_position = target_posisions[-1]
    target_last_direction = target_directions[-1]

    # 获取最新的飞机位置
    aircraft_last_position = aircraft_positions[-1]

    # 计算导弹末位置相对于飞机的位置向量
    relative_position = target_last_position - aircraft_last_position

    # 导弹方向与导弹到飞机的相对位置向量的点积
    is_ahead_of_target = np.dot(target_last_direction, relative_position) < 0

    if debug:
        reformat_array_for_print("target: ", target_posisions)
        reformat_array_for_print("aircraft: ", aircraft_positions)
        print("facing_target: ",facing_target[-1])
        print("target_posotion: ",target_position)
        print("is_ahead_of_target: ", is_ahead_of_target)

    return facing_target[-1], target_position, is_ahead_of_target


if __name__ == "__main__":
    # 导弹轨迹数据
    target_positions = np.array([
    [-2.4523915589009, 10015.171630321469, 2000.0101684140318],
    [-4.9073832091428935, 10030.35934616477, 2000.0424600862443],
    [-7.363291254364064, 10045.552737204158, 2000.095321057298],
    [-9.82059150238946, 10060.754863599035, 2000.1677037916625],
    [-12.27921248465505, 10075.965612188718, 2000.2598047810225],
    [-14.739076890037607, 10091.184898859074, 2000.3712714090461],
    [-17.20014802554816, 10106.412717107816, 2000.5006674545793],
    [-19.66248138288335, 10121.649125280917, 2000.64543537592],
    [-22.126299212765296, 10136.894229180236, 2000.8019172556724],
    [-24.592047325886295, 10152.14809829864, 2000.9659495341593],
    [-27.060450580266025, 10167.410729315845, 2001.1331748805997],
    [-29.5325348480894, 10182.682020210943, 2001.2994181864642],
    [-32.00959000280412, 10197.961645913738, 2001.4609979688457],
    [-34.493105237214294, 10213.249128590141, 2001.6150461615289],
    [-36.98467247635854, 10228.544048348263, 2001.7597679317532],
    [-39.48590163249683, 10243.846030434537, 2001.894335085839],
    [-41.99838646712327, 10259.154775938841, 2002.018724695271],
    [-44.52365607256576, 10274.470029652062, 2002.133502316463],
    [-47.063111614540496, 10289.791583600516, 2002.2395840851277],
    [-49.61800167446617, 10305.119260133983, 2002.3379959470003],
])

    # 飞机轨迹数据
    aircraft_positions = np.array([
    [0.012318679484371233, -9984.634054272097, 2000.0101676850609],
    [0.02465041956438956, -9969.251817004922, 2000.0424572521142],
    [0.03698670417415856, -9953.86391071129, 2000.0958278986363],
    [0.049330044955497636, -9938.467202294587, 2000.1687686249224],
    [0.06168035303439645, -9923.061802470722, 2000.2611283387241],
    [0.07403756743049322, -9907.64778717164, 2000.3720809276756],
    [0.08640170279081398, -9892.22513801409, 2000.4995485344134],
    [0.0987728227518035, -9876.79377553171, 2000.6403603382196],
    [0.11115100597789693, -9861.35360154934, 2000.7904774152485],
    [0.12353631892147737, -9845.90453316655, 2000.945162324154],
    [0.1359287870430959, -9830.446538654947, 2001.0992022676583],
    [0.14832836887225154, -9814.979669817922, 2001.247172522375],
    [0.1607349413863688, -9799.50408022691, 2001.3836557443256],
    [0.1731482952551469, -9784.020031154774, 2001.5034119451693],
    [0.1855681252331241, -9768.527903554734, 2001.60144991998],
    [0.19799398847689526, -9753.028250058927, 2001.672947457876],
    [0.21042542835094102, -9737.521640540956, 2001.713368548485],
    [0.22286192743075672, -9722.008720736032, 2001.71835852597],
    [0.23530290146192848, -9706.490219783138, 2001.6836159189224],
    [0.24774763008118403, -9690.967036638542, 2001.6045279820082],
])

    print(is_facing_target(target_positions, aircraft_positions, False))