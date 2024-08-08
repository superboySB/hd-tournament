import numpy as np
import torch

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

def estimate_direction_pytorch(positions):
    """ 使用 PyTorch 进行线性回归估计方向向量 """
    positions = torch.tensor(positions, dtype=torch.float32)
    t = torch.arange(positions.size(0), dtype=torch.float32).reshape(-1, 1)
    t_with_bias = torch.cat([t, torch.ones_like(t)], dim=1)

    # 使用伪逆求解线性回归参数，以防止奇异矩阵错误
    XTX_pinv = torch.linalg.pinv(t_with_bias.T.matmul(t_with_bias))
    XTy = t_with_bias.T.matmul(positions)
    params = XTX_pinv.matmul(XTy)
    direction = params[:-1].squeeze()  # 不需要截距

    norm = torch.linalg.norm(direction)
    if norm == 0:
        return direction.numpy()  # 如果方向向量的模为0，返回原向量
    return (direction / norm).numpy()

def is_facing_target(target_positions, aircraft_positions, debug=False):
    target_direction = estimate_direction_pytorch(target_positions)
    aircraft_direction = estimate_direction_pytorch(aircraft_positions)

    cos_theta = np.dot(target_direction, aircraft_direction)

    # 当两者方向的夹角大于90度时，认为是背对状态
    facing_target = cos_theta < 0

    # 预测目标的新位置，注意这里是减去500倍的速度向量
    target_position = target_positions[-1] - target_direction * 1000

    # 判断飞机是否在目标前方
    is_ahead_of_target = np.dot(target_direction, aircraft_positions[-1] - target_positions[-1]) > 0

    if debug:
        reformat_array_for_print("target: ", target_positions)
        reformat_array_for_print("aircraft: ", aircraft_positions)
        print("cos_theta: ", cos_theta)
        print("facing_target: ", facing_target)
        print("target_position: ", target_position)
        print("is_ahead_of_target: ", is_ahead_of_target)

    return facing_target, cos_theta, target_position, is_ahead_of_target


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

    print(is_facing_target(target_positions, aircraft_positions, True))