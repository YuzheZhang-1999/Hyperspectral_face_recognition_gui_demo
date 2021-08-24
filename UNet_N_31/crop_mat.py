import os
import os.path
import sys
import numpy as np
import cv2
from scipy.io import loadmat, savemat
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    """Perprocess dataset"""
    main_folder = 'S:/0.1 mosaic2hsi/高光谱-数据集生成/'
    data_orig_folder = os.path.join(main_folder, 'data/Data for Demosaicing-based-Spectrometer/ICVL/ICVL_train')
    data_folder = os.path.join(main_folder, 'results/UNet-hsi/train8-mat')
    img_folder = os.path.join(main_folder, 'results/UNet-hsi/train8-img')
    waste_folder = os.path.join(main_folder, 'results/UNet-hsi/waste8-mat')
    waste_img_folder = os.path.join(main_folder, 'results/UNet-hsi/waste8-img')
    mask_folder = os.path.join(main_folder, 'data/material and mask/mask/btes256_8channels.mat')
    material_folder = os.path.join(main_folder, 'data/material and mask/material/xiaoxiu8_evo3_64_100.mat')
    mask = np.asarray(loadmat(mask_folder)['mask'])
    material = np.asarray(loadmat(material_folder)['material_choose'])

    crop_sz = 256 # change mask_name!!!
    step = 128
    cont_var_thresh = 0.2
    freq_var_thresh = 50

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print('mkdir:{:s}'.format(data_folder))

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
        print('mkdir:{:s}'.format(img_folder))

    if not os.path.exists(waste_folder):
        os.makedirs(waste_folder)
        print('mkdir:{:s}'.format(waste_folder))

    if not os.path.exists(waste_img_folder):
        os.makedirs(waste_img_folder)
        print('mkdir:{:s}'.format(waste_img_folder))

    img_list = []
    for root, _, file_list in sorted(os.walk(data_orig_folder)):
        path = [os.path.join(root, x) for x in file_list]
        img_list.extend(path)

    for num, path in enumerate(img_list):
        print('processing: {}/{}'.format(num, len(img_list)))
        worker(material, mask, path, data_folder, waste_folder, img_folder, waste_img_folder, crop_sz, step, cont_var_thresh, freq_var_thresh)

    print('Done')

def worker(material, mask, path, data_folder, waste_folder, img_folder, waste_img_folder, crop_sz, step, cont_var_thresh, freq_var_thresh):
    img_name = os.path.basename(path)
    HSI = loadmat(path)
    HSI = np.asarray(HSI['rad'])

    h, w, c = HSI.shape

    h_sapce = np.arange(0, h - crop_sz + 1, step)
    w_sapce = np.arange(0, w - crop_sz + 1, step)

    idx = 0
    for x in h_sapce:
        for y in w_sapce:
            idx += 1

            patch_name = img_name.replace('.mat', '_s{:05d}.mat'.format(idx))
            img_patch_name = img_name.replace('.mat', '_s{:05d}.png'.format(idx))

            patch = HSI[x:x + crop_sz, y:y + crop_sz, :]
            patch_material_mod = np.dot(patch.reshape(crop_sz*crop_sz, -1), material[0:301:10, 1:])
            patch_mod = patch_material_mod*mask.reshape(crop_sz*crop_sz, -1)
            dataMat = np.dstack((patch_mod.reshape(crop_sz, crop_sz, -1), patch))
            dataMat = np.transpose(dataMat, [2, 0, 1])
            img_gray = np.uint16(patch[:, :, 1]*(2.**16))
            [mean, var] = cv2.meanStdDev(img_gray)
            var = var/mean
            freq_var = cv2.Laplacian(img_gray, cv2.CV_16U).mean()
            if var > cont_var_thresh and np.abs(freq_var) > freq_var_thresh:
                savemat(os.path.join(data_folder, patch_name), {'dataMat': dataMat}, do_compression=True)
                img_patch = np.delete(patch, 1, 2).astype(float)
                img_patch = img_patch ** (1 / 2.2) * 255.
                img_patch = np.uint8(np.clip(img_patch, 0, 255))
                cv2.imencode('.png', img_patch[:, :, 0:3])[1].tofile(os.path.join(img_folder, img_patch_name))
            else:
                savemat(os.path.join(waste_folder, patch_name), {'dataMat': dataMat}, do_compression=True)
                img_patch = np.delete(patch, 1, 2).astype(float)
                img_patch = img_patch ** (1 / 2.2) * 255.
                img_patch = np.uint8(np.clip(img_patch, 0, 255))
                cv2.imencode('.png', img_patch[:, :, 0:3])[1].tofile(os.path.join(waste_img_folder, img_patch_name))






























if __name__ == '__main__':
    main()