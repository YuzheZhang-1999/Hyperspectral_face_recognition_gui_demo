import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import skimage.measure as measure
from torch.utils.data import DataLoader
from model import UNet
import dataset_old as dataset
#import xlwt
from math import log10
import cv2
import sys
import numpy as np
sys.path.append('../')
from unet_rec_img.HSI2RGB import HSI2RGB

def main():
    dtype = 'float32'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.set_grad_enabled(False)

    # Net define
    MSnet = UNet.MultiscaleNet().to(device)
    MSnet.load_state_dict(torch.load('./checkpoint/checkpoint_withoutnorm_9_99.pt'))
    MSnet.eval()

    # training dataset#
    trainSet = dataset.MyDataSet('./Hsi_test_data_withnorm')
    trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    for i, data in enumerate(trainLoader, 0):
        trainmasaicPic, traingdPic, picName = data['masaic'].to(device),\
                                                  data['gd'].to(device),\
                                                  data['name'][0]
        print(trainmasaicPic)
        print(traingdPic)
        image_tensor = trainmasaicPic
        image_image = trainmasaicPic.squeeze().squeeze().cpu().numpy()
        #print(image_image)
        cv2.imshow('origin_image', image_image)
        hsi_image = MSnet(image_tensor)
        hsi_image = hsi_image.squeeze().permute(1, 2, 0)
        #print(hsi_image)
        #print(hsi_image.shape)
        hsi_rebuilt_image = HSI2RGB(hsi_image)
        hsi_rebuilt_image = (hsi_rebuilt_image.numpy())
        #print(hsi_rebuilt_image)
        cv2.imshow('hsi_rebuild', hsi_rebuilt_image)
        cv2.waitKey(500)

            

if __name__ == '__main__':
    main()
