import matplotlib.pyplot as plt
import numpy as np
"""
    Joint Operations for 3D Hand Pose Estimation
这是一个用于处理三维手部关节点建模、姿态估计、热力图可视化等任务的核心操作模块
    
"""

SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    3,
    0,  # 5's parent
    5,
    6,
    7,
    0,  # 9's parent
    9,
    10,
    11,
    0,  # 13's parent
    13,
    14,
    15,
    0,  # 17's parent
    17,
    18,
    19,
]


def rotation_matrix(yaw, pitch, roll):
    """
    Create a rotation matrix for the given yaw, pitch, and roll angles.
    Angles are given in radians.
    """
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    return Rz_yaw @ Ry_pitch @ Rx_roll


def calculate_point(yaw, pitch, roll, distance, current_position, current_orientation):
    """
    Calculate the global coordinates of a point after a rotation and translation.
    The rotation is specified by yaw, pitch, and roll angles, and the translation by distance.
    The calculation is based on the current position and orientation.
    """
    # Compute the rotation matrix and apply it to the current orientation
    R = rotation_matrix(yaw, pitch, roll)
    new_orientation = current_orientation @ R

    # Move in the direction of the new orientation
    direction = -new_orientation[:, 0]  # Assuming forward direction is the first column of the rotation matrix
    # direction /= np.linalg.norm(direction)
    new_position = current_position + direction * distance

    return new_position, new_orientation


def calculate_coords(angles, bone_lengths):
    num_joints = 21
    dimension = 3

    # Initialize joint coordinates
    joint_coords = np.zeros((num_joints, dimension))

    for finger_idx in range(5):
        current_position = np.array([0.0, 0.0, 0.0])
        current_orientation = np.eye(3)
        for joint_idx in range(4):
            current_idx = 4 * finger_idx + joint_idx

            flexion_angle, abduction_angle, twist_angle = angles[current_idx]
            distance = bone_lengths[current_idx]
            current_position, current_orientation = calculate_point(
                abduction_angle, flexion_angle, twist_angle, distance, current_position, current_orientation
            )
            joint_coords[current_idx + 1] = current_position

    return joint_coords

def calculate_angle(j1, j2, j3):
    """
    Calculate the angle between three points defined by joints j1, j2, and j3.
    The angle is calculated at joint j2.
    """
    v1 = j1 - j2
    v2 = j3 - j2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle


def calculate_pitch_roll(base_vector, target_vector):
    """
    Calculate the pitch and roll required to align the base vector with the target vector.
    """
    from scipy.spatial.transform import Rotation as R

    # Normalize the vectors
    base_vector_normalized = base_vector / np.linalg.norm(base_vector)
    target_vector_normalized = target_vector / np.linalg.norm(target_vector)

    # Calculate the rotation needed to align the base vector with the z-axis
    z_axis = np.array([0, 0, 1])
    rot_align_with_z = R.align_vectors([z_axis], [base_vector_normalized])[0]

    # Apply this rotation to the target vector
    rotated_target = rot_align_with_z.apply(target_vector_normalized)

    # Calculate pitch and roll from the rotated target vector
    # Pitch: rotation around y-axis, Roll: rotation around x-axis
    pitch = np.arcsin(rotated_target[2])  # z value
    roll = np.arctan2(-rotated_target[1], rotated_target[0])  # y and x values

    return pitch, roll


JOINT_ROOT_IDX = 9

REF_BONE_LINK = (0, 9)  # mid mcp

# bone indexes in 20 bones setting
ID_ROOT_bone = np.array([0, 4, 8, 12, 16])  # ROOT_bone from wrist to MCP
ID_PIP_bone = np.array([1, 5, 9, 13, 17])  # PIP_bone from MCP to PIP
ID_DIP_bone = np.array([2, 6, 10, 14, 18])  # DIP_bone from  PIP to DIP
ID_TIP_bone = np.array([3, 7, 11, 15, 19])  # TIP_bone from DIP to TIP

def angle_between(v1, v2):
    '''
    :param v1: B*3
    :param v2: B*3
    :return: B
    '''
    v1_u = normalize(v1.copy())
    v2_u = normalize(v2.copy())

    inner_product = np.sum(v1_u * v2_u, axis=-1)
    tmp = np.clip(inner_product, -1.0, 1.0)
    tmp = np.arccos(tmp)

    return tmp


def normalize(vec_):
    '''

    :param vec:  B*3
    :return:  B*1
    '''
    vec = vec_.copy()
    len = calcu_len(vec) + 1e-8

    return vec / len


def axangle2mat(axis, angle, is_normalized=False):
    '''

    :param axis: B*3
    :param angle: B*1
    :param is_normalized:
    :return: B*3*3
    '''
    if not is_normalized:
        axis = normalize(axis)

    x = axis[:, 0];
    y = axis[:, 1];
    z = axis[:, 2]
    c = np.cos(angle);
    s = np.sin(angle);
    C = 1 - c
    xs = x * s;
    ys = y * s;
    zs = z * s
    xC = x * C;
    yC = y * C;
    zC = z * C
    xyC = x * yC;
    yzC = y * zC;
    zxC = z * xC

    Q = np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])
    Q = Q.transpose(2, 0, 1)

    return Q


def calcu_len(vec):
    '''
    calculate length of vector
    :param vec: B*3
    :return: B*1
    '''

    return np.linalg.norm(vec, axis=-1, keepdims=True)


def caculate_ja(joint, vis=False):
    '''
    :param joint: 21*3
    :param vis:
    :return: 15*2
    '''
    ALL_bones = np.array([
        joint[i] - joint[SNAP_PARENT[i]]
        for i in range(1, 21)
    ])
    ROOT_bones = ALL_bones[ID_ROOT_bone]  # FROM THUMB TO LITTLE FINGER
    PIP_bones = ALL_bones[ID_PIP_bone]
    DIP_bones = ALL_bones[ID_DIP_bone]
    TIP_bones = ALL_bones[ID_TIP_bone]

    ALL_Z_axis = normalize(ALL_bones)
    PIP_Z_axis = ALL_Z_axis[ID_ROOT_bone]
    DIP_Z_axis = ALL_Z_axis[ID_PIP_bone]
    TIP_Z_axis = ALL_Z_axis[ID_DIP_bone]

    normals = normalize(np.cross(ROOT_bones[1:5], ROOT_bones[0:4]))

    # ROOT bones
    PIP_X_axis = np.zeros([5, 3])  # (5,3)
    PIP_X_axis[[0, 1, 4], :] = -normals[[0, 1, 3], :]
    PIP_X_axis[2:4] = -normalize(normals[2:4] + normals[1:3])
    PIP_Y_axis = normalize(np.cross(PIP_Z_axis, PIP_X_axis))

    tmp = np.sum(PIP_bones * PIP_Y_axis, axis=-1, keepdims=True)
    PIP_bones_xz = PIP_bones - tmp * PIP_Y_axis
    PIP_theta_flexion = angle_between(PIP_bones_xz, PIP_Z_axis)  # in global coordinate
    PIP_theta_abduction = angle_between(PIP_bones_xz, PIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((PIP_bones * PIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    PIP_theta_flexion[tmp_index] = -PIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((PIP_bones * PIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    PIP_theta_abduction[tmp_index] = -PIP_theta_abduction[tmp_index]

    temp_axis = normalize(np.cross(PIP_Z_axis, PIP_bones))
    temp_alpha = angle_between(PIP_Z_axis, PIP_bones)  # alpha belongs to [0, pi]
    temp_R = axangle2mat(axis=temp_axis, angle=temp_alpha, is_normalized=True)

    # DIP bones
    DIP_X_axis = np.matmul(temp_R, PIP_X_axis[:, :, np.newaxis])
    DIP_Y_axis = np.matmul(temp_R, PIP_Y_axis[:, :, np.newaxis])
    DIP_X_axis = np.squeeze(DIP_X_axis)
    DIP_Y_axis = np.squeeze(DIP_Y_axis)

    tmp = np.sum(DIP_bones * DIP_Y_axis, axis=-1, keepdims=True)
    DIP_bones_xz = DIP_bones - tmp * DIP_Y_axis
    DIP_theta_flexion = angle_between(DIP_bones_xz, DIP_Z_axis)  # in global coordinate
    DIP_theta_abduction = angle_between(DIP_bones_xz, DIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((DIP_bones * DIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    DIP_theta_flexion[tmp_index] = -DIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((DIP_bones * DIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    DIP_theta_abduction[tmp_index] = -DIP_theta_abduction[tmp_index]

    temp_axis = normalize(np.cross(DIP_Z_axis, DIP_bones))
    temp_alpha = angle_between(DIP_Z_axis, DIP_bones)  # alpha belongs to [pi/2, pi]
    temp_R = axangle2mat(axis=temp_axis, angle=temp_alpha, is_normalized=True)

    # TIP bones
    TIP_X_axis = np.matmul(temp_R, DIP_X_axis[:, :, np.newaxis])
    TIP_Y_axis = np.matmul(temp_R, DIP_Y_axis[:, :, np.newaxis])
    TIP_X_axis = np.squeeze(TIP_X_axis)
    TIP_Y_axis = np.squeeze(TIP_Y_axis)

    tmp = np.sum(TIP_bones * TIP_Y_axis, axis=-1, keepdims=True)
    TIP_bones_xz = TIP_bones - tmp * TIP_Y_axis
    TIP_theta_flexion = angle_between(TIP_bones_xz, TIP_Z_axis)  # in global coordinate
    TIP_theta_abduction = angle_between(TIP_bones_xz, TIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((TIP_bones * TIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    TIP_theta_flexion[tmp_index] = -TIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((TIP_bones * TIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    TIP_theta_abduction[tmp_index] = -TIP_theta_abduction[tmp_index]

    if vis:
        fig = plt.figure(figsize=[50, 50])
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 0],
                 joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 1],
                 joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 2], 'yo', label='keypoint')

        plt.plot(joint[:5, 0], joint[:5, 1],
                 joint[:5, 2],
                 '--y', )
        # label='thumb')
        plt.plot(joint[[0, 5, 6, 7, 8, ], 0], joint[[0, 5, 6, 7, 8, ], 1],
                 joint[[0, 5, 6, 7, 8, ], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 9, 10, 11, 12, ], 0], joint[[0, 9, 10, 11, 12], 1],
                 joint[[0, 9, 10, 11, 12], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 13, 14, 15, 16], 0], joint[[0, 13, 14, 15, 16], 1],
                 joint[[0, 13, 14, 15, 16], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 17, 18, 19, 20], 0], joint[[0, 17, 18, 19, 20], 1],
                 joint[[0, 17, 18, 19, 20], 2],
                 '--y',
                 )
        plt.plot(joint[4][0], joint[4][1], joint[4][2], 'rD', label='thumb')
        plt.plot(joint[8][0], joint[8][1], joint[8][2], 'r*', label='index')
        plt.plot(joint[12][0], joint[12][1], joint[12][2], 'r+', label='middle')
        plt.plot(joint[16][0], joint[16][1], joint[16][2], 'rx', label='ring')
        plt.plot(joint[20][0], joint[20][1], joint[20][2], 'ro', label='pinky')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        parent = np.array(SNAP_PARENT[1:])
        x, y, z = joint[parent, 0], joint[parent, 1], joint[parent, 2]
        u, v, w = ALL_bones[:, 0], ALL_bones[:, 1], ALL_bones[:, 2],
        ax.quiver(x, y, z, u, v, w, length=0.25, color="black", normalize=True)

        ALL_X_axis = np.stack((PIP_X_axis, DIP_X_axis, TIP_X_axis), axis=0).reshape(-1, 3)
        ALL_Y_axis = np.stack((PIP_Y_axis, DIP_Y_axis, TIP_Y_axis), axis=0).reshape(-1, 3)
        ALL_Z_axis = np.stack((PIP_Z_axis, DIP_Z_axis, TIP_Z_axis), axis=0).reshape(-1, 3)
        ALL_Bone_xz = np.stack((PIP_bones_xz, DIP_bones_xz, TIP_bones_xz), axis=0).reshape(-1, 3)

        ALL_joints_ID = np.array([ID_PIP_bone, ID_DIP_bone, ID_TIP_bone]).flatten()

        jx, jy, jz = joint[ALL_joints_ID, 0], joint[ALL_joints_ID, 1], joint[ALL_joints_ID, 2]
        ax.quiver(jx, jy, jz, ALL_X_axis[:, 0], ALL_X_axis[:, 1], ALL_X_axis[:, 2], length=0.05, color="r",
                  normalize=True)
        ax.quiver(jx, jy, jz, ALL_Y_axis[:, 0], ALL_Y_axis[:, 1], ALL_Y_axis[:, 2], length=0.10, color="g",
                  normalize=True)
        ax.quiver(jx, jy, jz, ALL_Z_axis[:, 0], ALL_Z_axis[:, 1], ALL_Z_axis[:, 2], length=0.10, color="b",
                  normalize=True)
        ax.quiver(jx, jy, jz, ALL_Bone_xz[:, 0], ALL_Bone_xz[:, 1], ALL_Bone_xz[:, 2], length=0.25, color="pink",
                  normalize=True)

        plt.legend()
        ax.view_init(-180, 90)
        ax.set_aspect('equal')
        plt.show()

    ALL_theta_flexion = np.stack((PIP_theta_flexion, DIP_theta_flexion, TIP_theta_flexion), axis=0).flatten()  # (15,)
    ALL_theta_abduction = np.stack((PIP_theta_abduction, DIP_theta_abduction, TIP_theta_abduction),
                                   axis=0).flatten()  # (15,)
    ALL_theta = np.stack((ALL_theta_flexion, ALL_theta_abduction), axis=1)  # (15, 2)

    return ALL_theta


class Angles2Joints:
    def __init__(self):
        self.angles = np.array([
            [ 15, -30.939622148340337, 10], # 1, Fixed
            [ (56 - 60), (-4.5 - 20), 120], # 2
            [ 5.8, 0, 0], # 3 
            [ -2.5,  0, 0], # 4
            [ 0, -15.570285130876561, 0], # 5, Fixed
            [ 10.8, -7.1, 0], # index
            [ 110.0, 0, 0], # 11
            [ 70.0, 0, 0], # 12
            [ 0, 0, 0], # 9, Fixed  
            [ 39.6, -5.5, 0], # middle
            [ 98.1, 0, 0], # 11
            [ 62.4, 0, 0], # 12
            [ 2.663503121773919, 12.44078179828681, 0], # 13, Fixed
            [ 57.6, -2.3, 0], # ring
            [ 94.2, 0, 0], # 11
            [ 59.9, 0, 0], # 12
            [ 6.242075032617077, 24.939622148340337, 0],# 17, Fixed
            [ 53.9, -3.6, 0], # pinky
            [ 100.6, 0, 0], # 11
            [ 64.0, 0, 0], # 12
        ])

        joint = np.array([[-0.6114823, 0.70412624, -0.36096475], # 0
                        [-0.38049987, 0.303464, -0.58379203], # 1
                        [-0.22365516, -0.05147859, -0.7509124], # 2
                        [-0.04204354, -0.33151808, -0.92656446], # 3
                        [0.09191624, -0.6089648, -1.0512774], # 4
                        [0.13345791, -0.02334917, -0.24654008], # 5
                        [0.34996158, -0.41111353, -0.16837479], # 6
                        [0.4534959, -0.69078416, -0.12496376], # 7
                        [0.49604133, -0.96323794, -0.10438757], # 8
                        [0., 0., 0.], # 9
                        [0.3559839, -0.37889394, 0.13638118], # 10
                        [0.48572803, -0.69607633, 0.13675757], # 11
                        [0.5390761, -0.9938516, 0.09033547], # 12
                        [-0.14901564, 0.00647402, 0.16235268], # 13
                        [0.19227624, -0.34850615, 0.29296255], # 14
                        [0.37767693, -0.57762665, 0.36711285], # 15
                        [0.27133223, -0.7816264, 0.20363101], # 16
                        [-0.3334502, 0.0463345, 0.27828288], # 17
                        [-0.2731263, -0.21098317, 0.49082187],
                        [-0.22576298, -0.40466458, 0.6499127],
                        [-0.16024478, -0.58365273, 0.8177859]])

        bone_lengths = np.array([np.linalg.norm(joint[i] - joint[SNAP_PARENT[i]]) for i in range(1, len(joint))])
        bone_lengths[0] = 0.467932775
        bone_lengths[1] = 0.467932775
        self.bone_lengths = bone_lengths


    def __call__(self, flexion_angles, abduction_angles):
        angles = self.angles
        angles[1, 0] = flexion_angles[0] - 60
        angles[1, 1] = abduction_angles[0] - 20
        angles[2, 0] = flexion_angles[1]
        angles[3, 0] = flexion_angles[2]

        for i in range(1, 5):
            angles[4 * i + 1, 0] = flexion_angles[3 * i]
            angles[4 * i + 1, 1] = abduction_angles[i]
            angles[4 * i + 2, 0] = flexion_angles[3 * i + 1]
            angles[4 * i + 3, 0] = flexion_angles[3 * i + 2]

        angles = np.deg2rad(angles)

        joint = calculate_coords(angles, self.bone_lengths)

        return joint
    

def generate_a_heatmap(arr, centers, max_values):
    """Generate pseudo heatmap for one keypoint in 3D.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: (img_h, img_w, img_d).
        centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: (M, 3).
        max_values (np.ndarray): The max values of each keypoint. Shape: (M,).

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """
    sigma = 1

    M, img_h, img_w, img_d = arr.shape

    for i, (center, max_value) in enumerate(zip(centers, max_values)):
        mu_x, mu_y, mu_z = center
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_z = max(int(mu_z - 3 * sigma), 0)
        ed_z = min(int(mu_z + 3 * sigma) + 1, img_d)

        y = np.arange(st_y, ed_y, 1, np.float32)
        x = np.arange(st_x, ed_x, 1, np.float32)
        z = np.arange(st_z, ed_z, 1, np.float32)

        if not (len(x) and len(y) and len(z)):
            continue

        y = y[:, None, None]
        x = x[None, :, None]
        z = z[None, None, :]

        patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2 + (z - mu_z)**2) / 2 / sigma**2)
        patch = patch * max_value
        arr[i, st_y:ed_y, st_x:ed_x, st_z:ed_z] = np.maximum(arr[i, st_y:ed_y, st_x:ed_x, st_z:ed_z], patch)

    return arr



def visualize_heatmap_3d(arr, threshold=0.4):
    """
    该函数用于可视化一个三维热力图（heatmap）的分布。
    将输入的多通道热力图在空间中投影并显示为三维散点图。
    """

    # 对所有通道求和，得到整体的空间热力图，形状变为(H, W, D)
    arr = arr.sum(0)

    # 创建一个新的画布
    fig = plt.figure()

    # 添加一个3D坐标系子图（projection='3d' 表示3维）
    ax = fig.add_subplot(111, projection='3d')

    # 获取热力图中大于阈值的点的坐标 (x, y, z)
    x, y, z = np.where(arr > threshold)

    # 提取这些点的强度值，用于颜色映射
    intensity = arr[x, y, z]

    # 使用散点图绘制这些点，并根据强度值进行颜色映射（使用 'viridis' 色彩）
    sc = ax.scatter(x, y, z, c=intensity, cmap='viridis')

    # 添加颜色条，表示颜色与强度之间的对应关系
    plt.colorbar(sc)

    # 设置坐标轴等比例，保证图形不失真
    ax.set_aspect('equal')

    # 设置视角角度（-180度俯仰角，90度方位角，即正视图）
    ax.view_init(-180, 90)

    # 显示图像
    plt.show()

