import scipy.io as sio
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

dtype = 'float32'


# data set
class MyDataSet(Dataset):
    def __init__(self, dataPath, origPath, transform=None):
        if not os.path.isdir(dataPath):
            print('Error: "', dataPath,
                  '" is not a directory or does not exist.')
            return
        self.picName = os.listdir(dataPath)
        self.picNum = len(self.picName)
        self.transform = transform
        self.dataPath = dataPath
        self.origPath = origPath

    def __len__(self):
        return self.picNum

    def __getitem__(self, idx):
        filePath = self.dataPath + '/' + self.picName[idx]
        masaic = sio.loadmat(filePath)['meas'].astype(dtype)
        masaic = masaic[np.newaxis, :, :]
        # plt.imshow(masaic, cmap='gray'), plt.show()
        pos, sceneName = sio.loadmat(filePath)['pos'].tolist(), sio.loadmat(filePath)['sceneName']

        gdPath = self.origPath + '/' + sceneName[0] + '.mat'
        Orig = sio.loadmat(gdPath)['rad'].astype(dtype)
        gd = Orig[pos[0][0]-1:pos[0][0]+pos[0][2]-1, pos[0][1]-1:pos[0][1]+pos[0][3]-1, :].transpose([2,0,1])
        # plt.imshow(gd[:,:,10], cmap='gray'), plt.show()
        sample = {'masaic': masaic, 'gd': gd, 'name': self.picName[idx]}
        if self.transform:
            sample.transform(sample)

        return sample

