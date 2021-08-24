import scipy.io as sio
import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

dtype = 'float32'


# data set
class MyDataSet(Dataset):
    def __init__(self, data_list, transform=None):
        super(MyDataSet, self).__init__()
        self.data_lists = []
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, idx):
        mask = sio.loadmat('../dataset/mask/btes256_5channels.mat')['mask'].astype(dtype)
        material = sio.loadmat('../dataset/material/xiaoxiu5_5.mat')['material_choose'][0:301:10, 1:].astype(dtype)
        tempData = sio.loadmat(self.data_lists[idx])['ps'].astype(dtype)
        h, w, c = tempData.shape
        temp = tempData.reshape(h*w, c)
        modData = np.dot(temp, material)
        modData = modData.reshape(h, w, -1)

        mosaic = np.zeros(modData.shape, dtype=dtype)
        for i in range(modData.shape[2]):
            mosaic[:, :, i] = np.multiply(modData[:, :, i], mask[:, :, i])

        mosaic = np.transpose(mosaic, [2, 0, 1])
        gd = np.transpose(tempData, [2, 0, 1])

        sample = {'masaic': mosaic, 'gd': gd, 'name': self.data_lists[idx]}

        return sample

