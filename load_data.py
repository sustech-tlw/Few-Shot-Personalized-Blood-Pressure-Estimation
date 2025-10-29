import torch
from torch.utils.data import Dataset
import numpy as np
from mat73 import loadmat
import scipy


class PPG_dataset(Dataset):
    def __init__(self, data_path, transforms=None, data_name='PulseDB_VitalDB', stage='pretrain'):
        # loading PPG and other information
        if data_name == 'PulseDB_VitalDB' and stage == 'pretrain':
            PPG, ABP = loading_data(data_path, data_name, stage)
            self.PPG = torch.from_numpy(PPG).float()
            self.data_name = data_name
            self.stage = stage
            self.ABP = torch.from_numpy(ABP).float()
        elif data_name == 'PulseDB_VitalDB' and stage == 'SBP':
            PPG, Label = loading_data(data_path, data_name, stage)
            self.PPG = torch.from_numpy(PPG).float()
            self.data_name = data_name
            self.stage = stage
            self.Label = torch.from_numpy(Label).float()
        elif data_name == 'PulseDB_VitalDB' and stage == 'DBP':
            PPG, Label = loading_data(data_path, data_name, stage)
            self.PPG = torch.from_numpy(PPG).float()
            self.data_name = data_name
            self.stage = stage
            self.Label = torch.from_numpy(Label).float()
        else:
            assert False, print('error data name or stage')

        self.transforms = transforms

    def __len__(self):
        return len(self.PPG)

    def __getitem__(self, idx):
        if self.transforms:
            if self.PPG.dim() == 3:
                PPG = self.transforms(self.PPG[idx, :, :])
            elif self.PPG.dim() == 2:
                PPG = self.transforms(self.PPG[idx, :])
        else:
            if self.PPG.dim() == 3:
                PPG = self.PPG[idx, :, :]
            elif self.PPG.dim() == 2:
                PPG = self.PPG[idx, :]
        if self.data_name == 'PulseDB_VitalDB' and self.stage == 'pretrain':
            ABP = self.ABP[idx, :]
            return PPG, ABP
        elif self.data_name == 'PulseDB_VitalDB' and self.stage == 'SBP':
            Label = self.Label[[idx]]
            return PPG, Label
        elif self.data_name == 'PulseDB_VitalDB' and self.stage == 'DBP':
            Label = self.Label[[idx]]
            return PPG, Label


class PPG_meta_dataset(Dataset):
    def __init__(self, data_path, data_name, stage, num_person=1293):
        if data_name == 'PulseDB_VitalDB' and (stage == 'SBP' or stage == 'DBP'):
            PPG, Label = loading_data(data_path, data_name, stage)
            num_PPG, c, t = PPG .shape
            assert num_PPG % num_person == 0, 'error: personal data split'
            PPG = PPG.reshape([num_person, -1, c, t])
            Label = Label.reshape([num_person, -1])
            self.data_name = data_name
            self.stage = stage
            self.PPG = PPG
            self.Label = torch.from_numpy(Label).float()

    def __len__(self):
        if self.data_name == 'PulseDB_VitalDB' and (self.stage == 'SBP' or self.stage == 'DBP'):
            return len(self.PPG)

    def __getitem__(self, idx):
        if self.data_name == 'PulseDB_VitalDB' and (self.stage == 'SBP' or self.stage == 'DBP'):
            per_PPG = self.PPG[idx]
            per_Label = self.Label[idx]
            return per_PPG, per_Label


def loading_data(data_path, data_name, stage):
    if data_name == 'PulseDB_VitalDB' and stage == 'pretrain':
        data = loadmat(data_path)
        Signals = data['Subset']['Signals'][:, :2, :]
        ABP = data['Subset']['Signals'][:, 2:, :]
        return Signals, ABP
    elif data_name == 'PulseDB_VitalDB' and stage == 'SBP':
        data = loadmat(data_path)
        Signals = data['Subset']['Signals'][:, :2, :]
        SBP = data['Subset']['SBP']
        return Signals, SBP
    elif data_name == 'PulseDB_VitalDB' and stage == 'DBP':
        data = loadmat(data_path)
        Signals = data['Subset']['Signals'][:, :2, :]
        DBP = data['Subset']['DBP']
        return Signals, DBP


if __name__ == '__main__':
    data_path = "E:\PluseDB\Supplementary_Subset_Files\VitalDB_CalFree_Test_Subset_no_filter.mat"
    dataset = PPG_dataset(data_path, data_name='PulseDB_VitalDB', stage='ft-SBP_multi_channel_sub_norm', num_sub=144)
    per_PPG, per_Label, Label_mean, Label_std = dataset[0]

