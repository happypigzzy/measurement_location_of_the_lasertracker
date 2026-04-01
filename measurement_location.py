# -*- coding: utf-8 -*-
import re
import numpy as np

# ==================== 将机器人坐标文本粘贴到此处 ====================
robot_text = """
    CONST robtarget Align_1:=[[867.84,-1533.52,1125.75],[0.555645,0.0945624,0.807106,-0.175777],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_2:=[[1517.51,-1177.01,518.70],[0.555635,0.0945654,0.807115,-0.175763],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_3:=[[1883.75,-747.48,366.25],[0.555631,0.0945712,0.807118,-0.175756],[-1,0,-1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_4:=[[2022.18,-1400.25,60.59],[0.555626,0.0945649,0.807123,-0.175755],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_5:=[[2187.60,-421.12,765.56],[0.555634,0.0945961,0.807113,-0.175758],[-1,0,-1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_6:=[[1955.28,20.13,209.63],[0.555628,0.0945874,0.807121,-0.175746],[0,0,-1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_7:=[[1378.60,14.98,-330.66],[0.555632,0.0946007,0.807115,-0.175753],[0,0,-1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_8:=[[1268.99,545.38,650.78],[0.55564,0.094593,0.807117,-0.175722],[0,0,-1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_9:=[[1800.09,-292.75,1117.73],[0.555695,0.0945735,0.807086,-0.175699],[-1,0,-1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Align_10:=[[2182.20,173.51,481.73],[0.55571,0.0945858,0.807077,-0.175689],[0,0,-1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    """

# ==================== 将激光数据文本粘贴到此处 ====================
tracker_text = """
1,4414.372,-1288.293,1717.813,0.8993,-0.2624,0.3499
2,3799.924,-1692.612,1103.64,0.8829,-0.3933,0.2564
3,3470.132,-2148.789,946.276,0.8282,-0.5129,0.2259
4,3283.63,-1507.454,640.235,0.8949,-0.4108,0.1745
5,3189.122,-2499.706,1341.159,0.7472,-0.5857,0.3142
6,3460.656,-2918.571,786.313,0.7532,-0.6352,0.1711
7,4039.519,-2865.495,252.32,0.8146,-0.5778,0.0509
8,4182.467,-3390.599,1232.18,0.7572,-0.6139,0.2231
9,3582.536,-2599.389,1697.224,0.7557,-0.5483,0.358
10,3243.918,-3090.445,1055.154,0.7047,-0.6714,0.2292
"""

def parse_robot_coords(text):
    """
    从 robtarget 定义文本中提取所有点的 [x, y, z] 坐标。
    匹配模式：[[数字,数字,数字]
    """
    pattern = r'\[\[([\d\.-]+),([\d\.-]+),([\d\.-]+)'
    matches = re.findall(pattern, text)
    coords = []
    for m in matches:
        x = float(m[0])
        y = float(m[1])
        z = float(m[2])
        coords.append([x, y, z])
    return np.array(coords)

def parse_tracker_coords(text):
    """
    从激光数据文本中提取所有点的 [x, y, z] 坐标。
    每行格式：id,x,y,z,i,j,k
    """
    lines = text.strip().splitlines()
    coords = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) >= 4:
            # 取第2,3,4个字段（索引1,2,3）
            x = float(parts[1].strip())
            y = float(parts[2].strip())
            z = float(parts[3].strip())
            coords.append([x, y, z])
    return np.array(coords)

# 解析坐标
robot_points = parse_robot_coords(robot_text)
tracker_points = parse_tracker_coords(tracker_text)

# 检查点数是否一致
print(f"机器人坐标点数: {len(robot_points)}")
print(f"激光坐标点数: {len(tracker_points)}")
if len(robot_points) != len(tracker_points):
    raise ValueError("机器人坐标点数和激光坐标点数不一致，请检查数据")

# ==================== 以下为刚体变换求解部分 ====================
def rigid_transform_3d(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t

def rotation_matrix_to_euler(R, seq='zyx'):
    import math
    if seq == 'zyx':
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            rx = math.atan2(R[2,1], R[2,2])
            ry = math.atan2(-R[2,0], sy)
            rz = math.atan2(R[1,0], R[0,0])
        else:
            rx = math.atan2(-R[1,2], R[1,1])
            ry = math.atan2(-R[2,0], sy)
            rz = 0
        return np.array([rx, ry, rz])
    else:
        raise ValueError("Unsupported Euler angle sequence")

R, t = rigid_transform_3d(robot_points, tracker_points)

T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t

print("\n===== 计算结果 =====")
print("旋转矩阵 R (机器人基座标系 → 追踪仪坐标系):")
print(R)
print("\n平移向量 t (机器人基座标系原点在追踪仪坐标系中的坐标):")
print(t)
print("\n齐次变换矩阵 (4x4):")
print(T)

euler_rad = rotation_matrix_to_euler(R, seq='zyx')
euler_deg = np.degrees(euler_rad)
print("\n欧拉角 (ZYX 顺序，即 Roll-Pitch-Yaw，单位: 度):")
print(f"Rx (绕 X): {euler_deg[0]:.4f}°")
print(f"Ry (绕 Y): {euler_deg[1]:.4f}°")
print(f"Rz (绕 Z): {euler_deg[2]:.4f}°")

transformed = (R @ robot_points.T).T + t
errors = transformed - tracker_points
rms = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
print(f"\n点集拟合均方根误差 (RMS): {rms:.6f} mm")