import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch

def HSI2RGB(img_hsi):
    hsi = img_hsi.cpu().numpy()
    rgb = sio.loadmat('../UNet_N_31/rgb.mat')['rgb'] # 读取rgb曲线
    h, w, c = hsi.shape
    img_rgb = np.zeros([h, w, 3])
    for i in range(3):
        img_rgb[:, :, i] = np.sum(hsi * rgb[:, i], -1)  # img_rgb为合成后的彩色图像
    img_rgb = torch.from_numpy(img_rgb)
    return img_rgb



# plt.imshow(img_rgb/10)
# plt.show()
