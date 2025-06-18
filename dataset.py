from torch.utils.data import Dataset
import numpy as np
import torch
import os

from joint_ops import Angles2Joints


class EMGDataset(Dataset):

    def __init__(self, root_dir):
        super(EMGDataset, self).__init__()
        self.emg = np.load(os.path.join(root_dir, 'emg_train.npy'))
        self.force = np.load(os.path.join(root_dir, 'force_train.npy'))
        self.force_class = np.load(os.path.join(root_dir, 'force_class_train.npy'))

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        emg = self.emg[idx]
        force = self.force[idx]
        force_class = self.force_class[idx]
        return torch.from_numpy(emg), torch.from_numpy(force), torch.from_numpy(force_class)


class EMGDataset3DPose(Dataset):

    def __init__(self, root_dir):
        super(EMGDataset3DPose, self).__init__()
        self.emg = np.load(os.path.join(root_dir, 'emg_train.npy'))
        self.force = np.load(os.path.join(root_dir, 'force_train.npy'))
        self.force_class = np.load(os.path.join(root_dir, 'force_class_train.npy'))
        self.angles2joints = Angles2Joints()

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        emg = self.emg[idx, :8]
        pose = self.emg[idx, 8:, -1]
        angles_spread = pose[::4]
        angles_stretch = pose[np.arange(pose.shape[0]) % 4 != 0]

        force = self.force[idx]
        force_class = self.force_class[idx]
        return torch.from_numpy(emg), torch.from_numpy(angles_spread), torch.from_numpy(angles_stretch), torch.from_numpy(force), torch.from_numpy(force_class)
