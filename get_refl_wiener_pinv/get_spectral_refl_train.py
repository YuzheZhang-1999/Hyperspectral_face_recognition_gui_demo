import os
import xlwt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    print('get spectral reflectivity!')
    spectral_dir = './database_spectral_curve_excel'
    train_refl_txt_path = './train_txt_resp_refl/train_spectral_refl.txt'
    data_list_index = [1054, 1100, 1145, 1191, 1236, 1282, 1327, 1372, 1417, 1463,1507, 1551, 1596, 1640, 1685, 1729, 1773, 1817, 1861, 1905,1948, 1992, 2035, 2078, 2122, 2166, 2209, 2252, 2295, 2338,2381]
    spectral_data_num = len(os.listdir(spectral_dir))
    all_spectral_data = np.zeros((spectral_data_num, 31))

    for root_dir, sub_dir, files in os.walk(spectral_dir):
        for curve_num, file in enumerate(files):
            if file.endswith('.csv'):
                file_name = os.path.join(root_dir, file)
                print(file_name)
                spectral_raw_data = np.array(pd.read_csv(file_name, error_bad_lines=False))
                for singel_curve_dot, index in enumerate(data_list_index):
                    all_spectral_data[curve_num, singel_curve_dot] = spectral_raw_data[index, 1] if spectral_raw_data[index, 1] > 0 else 0
    np.savetxt(train_refl_txt_path, all_spectral_data, fmt="%f", delimiter=',')
    print('用于训练的数据保存在了： ' + train_refl_txt_path)

