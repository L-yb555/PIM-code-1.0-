import torch
import numpy as np
import matplotlib.pyplot as plt

'''
对比维度
             joint_ops.py    joint_ops_torch.py
是否支持 GPU     ❌ 否            ✅ 是 (cuda)
是否支持 Batch      ❌ 否         ✅ 是
是否可导            ❌ 否            ✅ 是
数据类型       NumPy arrays       PyTorch tensors
是否适合模型集成       ❌ 否       ✅ 是
是否支持自动微分       ❌ 否       ✅ 是
主要用途       调试、可视化       模型训练、推理

'''




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

class Angles2Joints(torch.nn.Module):
    def __init__(self):
        super().__init__()
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

        # Convert numpy arrays to PyTorch tensors
        bone_lengths = np.array([np.linalg.norm(joint[i] - joint[SNAP_PARENT[i]]) for i in range(1, len(joint))])
        bone_lengths[0] = 0.467932775
        bone_lengths[1] = 0.467932775
        self.bone_lengths = torch.tensor(bone_lengths, dtype=torch.float32, device='cuda')
        self.angles = torch.tensor(self.angles, dtype=torch.float32)

    def __call__(self, flexion_angles, abduction_angles):
        # Batch size
        B = flexion_angles.shape[0]

        # Process angles
        angles = self.angles.unsqueeze(0).repeat(B, 1, 1)  # Replicating for each batch
        angles[:, 1, 0] = flexion_angles[:, 0] - 60
        angles[:, 1, 1] = abduction_angles[:, 0] - 20
        angles[:, 2, 0] = flexion_angles[:, 1]
        angles[:, 3, 0] = flexion_angles[:, 2]

        # Process for each finger
        for i in range(1, 5):
            angles[:, 4 * i + 1, 0] = flexion_angles[:, 3 * i]
            angles[:, 4 * i + 1, 1] = abduction_angles[:, i]
            angles[:, 4 * i + 2, 0] = flexion_angles[:, 3 * i + 1]
            angles[:, 4 * i + 3, 0] = flexion_angles[:, 3 * i + 2]
        angles = torch.deg2rad(angles)
        joint = self.calculate_coords(angles, self.bone_lengths)

        return joint

    @staticmethod
    def rotation_matrix(yaw, pitch, roll):
        # Create batched rotation matrices
        Rz_yaw = torch.stack([
            torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),
            torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
            torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
        ], dim=-1).cuda().reshape(-1, 3, 3)

        Ry_pitch = torch.stack([
            torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch),
            torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
            -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)
        ], dim=-1).cuda().reshape(-1, 3, 3)

        Rx_roll = torch.stack([
            torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
            torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
            torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)
        ], dim=-1).cuda().reshape(-1, 3, 3)

        return Rz_yaw @ Ry_pitch @ Rx_roll

    @staticmethod
    def calculate_point(yaw, pitch, roll, distance, current_position, current_orientation):
        # Calculate the global coordinates of points for all batches
        R = Angles2Joints.rotation_matrix(yaw, pitch, roll)
        new_orientation = torch.matmul(current_orientation, R)

        direction = -new_orientation[:, :, 0]
        new_position = current_position + direction * distance.unsqueeze(-1)

        return new_position, new_orientation

    def calculate_coords(self, angles, bone_lengths):
        B, num_joints, _ = angles.shape

        # Initialize joint coordinates for all batches
        joint_coords = torch.zeros((B, num_joints + 1, 3), dtype=torch.float32, device='cuda')
        current_position = torch.zeros((B, 5, 3), dtype=torch.float32, device='cuda')  # for 5 fingers
        current_orientation = torch.eye(3, device='cuda').unsqueeze(0).unsqueeze(0).repeat(B, 5, 1, 1)  # for 5 fingers

        # Reshape angles for batch operations
        angles = angles.view(B, 5, 4, 3)  # Reshape to (Batch, Fingers, Joints, Angles)

        for joint_idx in range(4):
            flexion_angle = angles[:, :, joint_idx, 0]
            abduction_angle = angles[:, :, joint_idx, 1]
            twist_angle = angles[:, :, joint_idx, 2]
            distance = bone_lengths.view(5, 4)[:, joint_idx]

            # Calculate rotation matrices for all joints in all fingers simultaneously
            R = self.rotation_matrix(abduction_angle, flexion_angle, twist_angle)
            R = R.view(current_orientation.size())

            # Update current orientation
            current_orientation = torch.matmul(current_orientation, R)

            direction = -current_orientation[:, :, :, 0]
            new_position = current_position + direction * distance.unsqueeze(-1).unsqueeze(0)

            # Update positions
            joint_coords[:, 1 + joint_idx::4, :] = new_position.view(B, -1, 3)
            current_position = new_position

        return joint_coords


def generate_a_heatmap(arr_shape, centers):
    """Generate pseudo heatmap for multiple keypoints in 3D with batch processing.

    Args:
        arr_shape (tuple): The array shape to store the generated heatmaps. (M, img_h, img_w, img_d).
        centers (torch.Tensor): The coordinates of corresponding keypoints (of multiple persons in a batch). 
                                Shape: (B, M, 3).

    Returns:
        torch.Tensor: The generated pseudo heatmaps.
    """
    sigma = 1

    B = centers.size(0)
    M, img_h, img_w, img_d = arr_shape
    arr = torch.zeros((B, *arr_shape), dtype=torch.float32, device='cuda')
    max_values = torch.ones((B, M, 1, 1, 1), device='cuda')

    # Creating meshgrids
    y = torch.arange(0, img_h, dtype=torch.float32, device='cuda')
    x = torch.arange(0, img_w, dtype=torch.float32, device='cuda')
    z = torch.arange(0, img_d, dtype=torch.float32, device='cuda')
    yy, xx, zz = torch.meshgrid(y, x, z, indexing='ij')

    # Reshaping and broadcasting centers for vectorized computation
    centers = centers.view(B, M, 1, 1, 1, 3)
    mu_x, mu_y, mu_z = centers[..., 0], centers[..., 1], centers[..., 2]

    # Calculating the heatmap
    patch = torch.exp(-((xx - mu_x)**2 + (yy - mu_y)**2 + (zz - mu_z)**2) / (2 * sigma**2))
    patch *= max_values
    arr = torch.maximum(arr, patch)

    return arr


def visualize_heatmap_3d(arr, threshold=0.4):
    arr = arr.sum(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # arr 배열에서 임계값 이상인 요소의 좌표를 가져옵니다.
    x, y, z = np.where(arr > threshold)

    # 히트맵 강도를 색상으로 표현합니다.
    intensity = arr[x, y, z]

    # 3D 스캐터 플롯으로 시각화합니다.
    sc = ax.scatter(x, y, z, c=intensity, cmap='viridis')

    # 컬러바를 추가합니다.
    plt.colorbar(sc)
    ax.set_aspect('equal')
    ax.view_init(-180, 90)
    plt.show()


class Angles2Joints_Redefined(Angles2Joints):
    def __init__(self):
        super().__init__()
        self.angles = np.array([
            [ 30, -50.939622148340337, 40], # 1, Fixed
            [ (48.4 - 60), (57.1 - 20), -20], # 2
            [ 60.0, 0, 0], # 3 
            [ 26.7,  0, 0], # 4
            [ 0, -15.570285130876561, 0], # 5, Fixed
            [ 60.1, 1.3, 0], # index
            [ 31.1, 0, 0], # 11
            [ 15.1, 0, 0], # 12
            [ 0, 0, 0], # 9, Fixed  
            [ 72.1, -2.1, 0], # middle
            [ 33.3, 0, 0], # 11
            [ 16.3, 0, 0], # 12
            [ 2.663503121773919, 12.44078179828681, 0], # 13, Fixed
            [ 68.0, -9.5, 0], # ring
            [ 35.1, 0, 0], # 11
            [ 20.1, 0, 0], # 12
            [ 6.242075032617077, 24.939622148340337, 0],# 17, Fixed
            [ 64.4, -6.2, 0], # pinky
            [ 40.0, 0, 0], # 11
            [ 25.0, 0, 0], # 12
        ])
        self.angles = torch.tensor(self.angles, dtype=torch.float32)