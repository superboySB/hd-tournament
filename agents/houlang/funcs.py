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

def is_facing_missile(missile_positions, aircraft_positions, debug=False):
    # 时间间隔
    delta_t = 0.05

    # 计算速度向量（方向）
    aircraft_directions = (aircraft_positions[1:] - aircraft_positions[:-1]) / delta_t
    missile_directions = (missile_positions[1:] - missile_positions[:-1]) / delta_t

    # 归一化方向向量
    aircraft_directions /= np.linalg.norm(aircraft_directions, axis=1)[:, None]
    missile_directions /= np.linalg.norm(missile_directions, axis=1)[:, None]

    # 计算相对位置向量
    relative_positions = missile_positions[:-1] - aircraft_positions[:-1]

    # 计算导弹相对位置的单位方向向量
    relative_directions = relative_positions / np.linalg.norm(relative_positions, axis=1)[:, None]

    # 计算飞机和导弹的夹角余弦值
    cos_theta = np.sum(aircraft_directions * relative_directions, axis=1)
    facing_missile = cos_theta > 0

    # 计算目标位置
    target_position = missile_positions[-1] - missile_directions[-1] * 500

    # 判断飞机是否在导弹的前方
    # 获取最新的导弹位置和方向
    missile_last_position = missile_positions[-1]
    missile_last_direction = missile_directions[-1]

    # 获取最新的飞机位置
    aircraft_last_position = aircraft_positions[-1]

    # 计算导弹末位置相对于飞机的位置向量
    relative_position = missile_last_position - aircraft_last_position

    # 导弹方向与导弹到飞机的相对位置向量的点积
    is_ahead_of_enemy = np.dot(missile_last_direction, relative_position) < 0

    if debug:
        # reformat_array_for_print("missile: ", missile_positions)
        # reformat_array_for_print("aircraft: ", aircraft_positions)
        print("facing_missile: ",facing_missile[-1])
        print("target_posotion: ",target_position)
        print("is_ahead_of_enemy: ", is_ahead_of_enemy)

    return facing_missile[-1], target_position, is_ahead_of_enemy


if __name__ == "__main__":
    # 导弹轨迹数据
    missile_positions = np.array([
    [6203.956534419035, -21886.617932266006, -666.535887889814],
    [6214.8164832309185, -21864.532640215504, -686.8937543556983],
    [6200.347360680451, -21804.101627675285, -684.855907116984],
    [6198.347117589384, -21779.754141956106, -689.203814013242],
    [6202.475283334118, -21741.80941447221, -717.5140834152978],
    [6198.860410547133, -21715.157491118633, -709.320376881177],
    [6163.50624540508, -21671.165613530455, -710.2024553590281],
    [6166.446947757104, -21629.296721085888, -704.343982513366],
    [6167.1512952361745, -21599.609760365427, -725.1044563294909],
    [6181.653005574025, -21549.436044490085, -743.8742634095317],
    [6163.655521939539, -21507.07857527548, -730.457906641669],
    [6175.012855881755, -21465.425155674067, -743.198754769059],
    [6154.293618903804, -21433.919089188712, -737.6122169063945],
    [6164.250397048205, -21399.84187101236, -775.6601582666352],
    [6171.245815466935, -21366.01343923536, -776.5202459641251],
    [6143.432315143546, -21325.085690988286, -769.8639808701134],
    [6136.165984197096, -21275.696981242654, -785.4895832464043],
    [6133.57232818609, -21230.425365166968, -768.3198452692394],
    [6142.694297148501, -21208.653206965882, -765.8019922886656],
    [6136.648731741107, -21163.168521808442, -792.4812763348184],
])

    # 飞机轨迹数据
    aircraft_positions = np.array([
    [1845.2542926067385, -4788.821201502154, -1469.9979350481062],
    [1844.4064377940463, -4783.70078705535, -1473.3354046532331],
    [1843.550985103312, -4778.583421056879, -1476.6654488894674],
    [1842.6878170763325, -4773.469188298835, -1479.9880784699544],
    [1841.816813981202, -4768.358182359973, -1483.3033074620107],
    [1840.9378543575394, -4763.250504145482, -1486.6111523106283],
    [1840.0508155276461, -4758.146260565036, -1489.9116309891378],
    [1839.1555739243645, -4753.0455637244495, -1493.204762387193],
    [1838.252005088391, -4747.948531154827, -1496.4905661733774],
    [1837.3399838865207, -4742.855285471214, -1499.7690623421586],
    [1836.4193849935032, -4737.765953336682, -1503.0402704621974],
    [1835.4900835630885, -4732.680663928617, -1506.304208725971],
    [1834.551956029347, -4727.599547019399, -1509.5608929210339],
    [1833.6048809881179, -4722.522730788433, -1512.8103353615406],
    [1832.6487401213712, -4717.450339459545, -1516.0525438685981],
    [1831.683419101709, -4712.382490879116, -1519.2875208516725],
    [1830.7088083793203, -4707.319294336803, -1522.515262606863],
    [1829.724803895963, -4702.260848474757, -1525.7357587775623],
    [1828.7313076318217, -4697.207239546091, -1528.9489920501474],
    [1827.7282279638878, -4692.15854010557, -1532.1549381096956],
])

    print(is_facing_missile(missile_positions, aircraft_positions, False))