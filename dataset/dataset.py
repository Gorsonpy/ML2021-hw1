import csv

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class COVIDDataset(Dataset):
    # target_only: 仅使用target强相关的特征
    def __init__(self, path, mode = 'train', target_only = False):
        self.mode = mode

        with open(path, 'r') as f:
            data = list(csv.reader(f))
            # 从第二行开始读取数据，第一行是表头
            data = np.array(data[1:])[:, 1:].astype(np.float32) # 去除行头和列头


        if not target_only:
            feats = list(range(93))
        else:
            feats = list(range(40)) + [57, 75]

        if mode == 'test':
            data = data[:, feats] # feats是特征的索引列表，因为有可能不是连续的是前40 + 57 + 75
            self.data = torch.tensor(data, dtype = torch.float32)
        else :
            target = data[:, -1] # 输出
            data = data[:, feats] # 输入特征
            self.data = torch.tensor(data, dtype = torch.float32)
            self.target = torch.tensor(target, dtype = torch.float32)

            # 从数据集中划分训练集和验证集
            if mode == 'train':
                idxs = [i for i in range(len(self.data)) if i % 10 != 0]
            elif mode == 'valid':
                idxs = [i for i in range(len(self.data)) if i % 10 == 0]

        # normalization
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim = 0, keepdim = True)) / self.data[:, 40:].std(dim = 0, keepdim = True)
        self.dim = self.data.shape[1] # self.data.shape = (data_num, dim), dim是特征维度，data_num是数据个数

        print("Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})".format(mode, len(self.data), self.dim))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.target[idx]) if self.mode != 'test' else self.data[idx]

def covid_dataloader(path, batch_size, mode = 'train', target_only = False, n_job = 0):
    dataset = COVIDDataset(path, mode = mode, target_only = target_only)
    # pin_memory = True: 将数据保存在固定内存中，加速GPU读取
    # num_workers = n_job: 使用多线程加载数据
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (mode == 'train'), num_workers = n_job, pin_memory = True)
    return dataloader