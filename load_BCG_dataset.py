from scipy.io import loadmat
import numpy as np
from mat73 import loadmat
from torch.utils.data import Dataset
import Util.NoiseGenerator as noisy

class BCG_dataset(Dataset):
    def __init__(self, data_path, data_name, stage, add_noisy=False):
        data = loadmat(data_path)
        self.PPG = data['PPG_list']
        self.PPG_IMFs = data['PPG_IMFs_list']
        self.central_fs = data['central_fs_list']
        self.SBP = data['SBP_list']
        self.DBP = data['DBP_list']
        self.data_name = data_name
        self.stage = stage
        self.add_noisy = add_noisy

    def __len__(self):
        return len(self.PPG)

    def __getitem__(self, index):
        if self.stage == 'ft-SBP':
            PPG = self.PPG[index]
            if self.add_noisy == True:
                PPG = np.vstack([noisy.add_motion_artifact(PPG[i, :], fs=125, alpha=0.5, smr=0, dur_range_ms=[1000, 2000]) for i in range(len(PPG))])
            PPG = PPG[:, np.newaxis, :]
            Label = self.SBP[index]
            return PPG, Label
        elif self.stage == 'ft-DBP':
            PPG = self.PPG[index]
            if self.add_noisy == True:
                PPG = np.vstack([noisy.add_motion_artifact(PPG[i, :], fs=125, alpha=0.5, smr=0, dur_range_ms=[1000, 2000]) for i in range(len(PPG))])
            PPG = PPG[:, np.newaxis, :]
            Label = self.DBP[index]
            return PPG, Label
        else:
            # 根据场景返回其他数据或抛出明确错误
            raise ValueError(f"Unsupported stage: {self.stage}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_path = "E:\BCG\\bcg_dataset\signal_all_subject_VMD.mat"
    data_set = BCG_dataset(data_path, data_name='BCG_VMD', stage='ft-SBP', add_noisy=True)
    PPG_IMFs, Label = data_set[0]
    print(PPG_IMFs.shape)