import os
import pandas as pd
import numpy as np
import cv2

if __name__ == '__main__':
    print('get camera response!')
    image_path_dir = './database_mono_image'
    train_resp_txt_path = './train_txt_resp_refl/train_camera_resp.txt'
    pic_num = len(os.listdir(image_path_dir))
    pic_feature_axis = [[1024,800],[1185,800],[1355,800],[1535,800],[1024,800],[1185,970],[1355,970],[1535,970],[1535,110]]
    camera_resp = np.zeros((pic_num, 9))
    for root_dir, sub_dir, files in os.walk(image_path_dir):
        for pic_count, file in enumerate(files):
            file_name = os.path.join(root_dir, file)
            print(file_name)
            image_array = cv2.imread(file_name)
            for i in range(9):
                camera_resp[pic_count, i] = image_array[pic_feature_axis[i][0], pic_feature_axis[i][1], 0]
    camera_resp = camera_resp/255
    np.savetxt(train_resp_txt_path, camera_resp, fmt="%f", delimiter=',')
    print('用于训练的数据保存在了： ' + train_resp_txt_path)
