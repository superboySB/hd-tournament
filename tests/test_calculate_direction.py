import math

def normalize_angle(angle):
    """ 将角度归一化到 [-π, π] 区间。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def calculate_direction(plane_info, target_info, debug=True):
    dx = target_info.x - plane_info.x
    dy = target_info.y - plane_info.y
    dz = target_info.z - plane_info.z
    horizontal_distance = math.sqrt(dx ** 2 + dy ** 2)
    azimuth = math.atan2(dy, dx)

    if debug:
        print(f"\ndx: {dx}, dy: {dy}, dz: {dz}")
        print(f"horizontal_distance: {horizontal_distance}")
        print(f"Calculated azimuth (before normalization): {azimuth}, Current yaw: {plane_info.yaw}")

    # 规范化所有角度
    azimuth = normalize_angle(azimuth)
    current_yaw = normalize_angle(plane_info.yaw)

    # 计算角度差，并确保始终取最短路径
    azimuth_diff = normalize_angle(azimuth - current_yaw)

    if debug:
        print(f"Normalized azimuth: {azimuth}, Normalized current yaw: {current_yaw}")
        print(f"Azimuth difference: {azimuth_diff}")

    return azimuth, azimuth_diff

# 假设的飞机和目标信息
class Info:
    def __init__(self, x, y, z, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw

# 示例数据
plane_info = Info(0, 0, 0, 1.571)  # 假设当前偏航角为 90 度 (π/2)
target_info = Info(8000, 43120, -186, 0)  # 目标位置

azimuth, azimuth_diff = calculate_direction(plane_info, target_info, debug=True)
print(f"Final azimuth: {azimuth}, Azimuth difference: {azimuth_diff}")
