import cv2
import numpy as np
import scipy.io as scio
import get_refl_wiener_pinv.mian_wiener_ref_rec as get_refl
import json

camera_blocks_multi_axis = [
    [[353, 193], [495, 178], [665, 175], [856, 162], [335, 370], [495, 365], [668, 343], [836, 331],
     [848, 491], ],
    [[981, 169], [1130, 160], [1313, 158], [1510, 145], [985, 330], [1165, 328], [1318, 310], [1512, 300],
     [1512, 453], ],
    [[1648, 125], [1809, 126], [1977, 120], [2143, 106], [1649, 305], [1815, 286], [1991, 285], [2138, 271],
     [2156, 436], ],
    [[369, 829], [527, 831], [699, 831], [891, 820], [384, 1012], [530, 1011], [713, 1008], [877, 983],
     [895, 1144], ],
    [[1024, 800], [1185, 800], [1355, 800], [1535, 800], [1024, 800], [1185, 970], [1355, 970], [1535, 970],
     [1535, 110], ],
    [[1667, 753], [1838, 755], [2014, 745], [2163, 741], [1675, 951], [1837, 957], [2029, 940], [2178, 919],
     [2181, 1072], ],
    [[401, 1496], [557, 1513], [747, 1497], [924, 1494], [435, 1659], [569, 1646], [737, 1642], [905, 1635],
     [924, 1785], ],
    [[1063, 1475], [1200, 1470], [1366, 1455], [1559, 1455], [1061, 1632], [1207, 1625], [1394, 1611], [1552, 1593],
     [1566, 1754], ],
    [[1696, 1445], [1869, 1440], [2051, 1427], [2210, 1421], [1719, 1608], [1882, 1594], [2068, 1583], [2203, 1552],
     [2217, 1702], ],
]


def load_multi_textures_data(texture_name, texture_refl_array):
    texture_json_file = '../other_utils/json_database/multi_textures_detect_database.json'
    new_data = {texture_name: texture_refl_array}
    try:
        with open(texture_json_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
            load_data = old_data
            load_data.update(new_data)
    except:
        load_data = new_data
    # load_json_data = json.dumps(load_data, indent=4)
    load_json_data = json.dumps(load_data, sort_keys=True, indent=4)
    with open(texture_json_file, 'w', encoding='utf-8') as texture_file:
        texture_file.write(load_json_data)


def get_refl_wiener_with_resp_list(camera_resp_list):
    train_camera_resp = '../get_refl_wiener_pinv/train_txt_resp_refl/train_camera_resp.txt'
    train_spectral_refl = '../get_refl_wiener_pinv/train_txt_resp_refl/train_spectral_refl.txt'
    resp_train = np.loadtxt(train_camera_resp, dtype=np.float64, delimiter=',')
    refl_train = np.loadtxt(train_spectral_refl, dtype=np.float64, delimiter=',')

    camera_refl = get_refl.func_wiener_ref_rec(resp_train, refl_train, camera_resp_list)
    # print('get test refl curve!')
    return camera_refl


# num block
# 相机共有9个block，每个block包含一个完整的9通道的滤光片
# 0 1 2
# 3 4 5
# 6 7 8


def get_texture_name_with_num_block(camera_refl):
    texture_json_file = '../other_utils/json_database/multi_textures_detect_database.json'
    texture_scores_dict = {}
    with open(texture_json_file, 'r', encoding='utf-8') as texture_json_file:
        texture_database = json.load(texture_json_file)
        for texture_key in texture_database.keys():
            texture_feature = texture_database[texture_key]
            texture_sim_score = np.dot(camera_refl, texture_feature) / (
                        np.linalg.norm(camera_refl) * np.linalg.norm(texture_feature))
            texture_scores_dict[texture_key] = texture_sim_score
    texture_sorted_scores_dict = sorted(texture_scores_dict.items(), key=lambda x: x[1], reverse=True)
    return texture_sorted_scores_dict[0]


def get_camera_resp_matrix_with_num_block(input_image, num_block):
    pic_feature_axis = camera_blocks_multi_axis[num_block]
    camera_resp = np.zeros((1, 9))
    try:
        image_array = input_image[:, :, 0]
    except:
        image_array = input_image
    for i in range(9):
        camera_resp[0, i] = image_array[pic_feature_axis[i][1], pic_feature_axis[i][0]]
    camera_resp = camera_resp / 255.0
    return camera_resp


def load_multi_camera_resp_data(camera_resp_object_name, camera_resp_array):
    camera_resp_json_file = '../other_utils/json_database/multi_camera_resp_database.json'
    new_data = {camera_resp_object_name: camera_resp_array}
    try:
        with open(camera_resp_json_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
            load_data = old_data
            load_data.update(new_data)
    except:
        load_data = new_data
    # load_json_data = json.dumps(load_data, indent=4)
    load_json_data = json.dumps(load_data, sort_keys=True, indent=4)
    with open(camera_resp_json_file, 'w', encoding='utf-8') as f:
        f.write(load_json_data)

# num block
# 相机共有9个block，每个block包含一个完整的9通道的滤光片
# 0 1 2
# 3 4 5
# 6 7 8


def get_texture_from_camera_resp_with_num_block(resp_test, num_block):
    camera_resp_json_file = '../other_utils/json_database/multi_camera_resp_database.json'
    texture_scores_from_resp_dict = {}
    with open(camera_resp_json_file, 'r', encoding='utf-8') as f:
        camera_resp_database = json.load(f)
        for camera_resp_key in camera_resp_database.keys():
            camera_resp_list = np.asarray(camera_resp_database[camera_resp_key])[num_block]
            texture_sim_score = np.linalg.norm(resp_test - camera_resp_list)
            texture_scores_from_resp_dict[camera_resp_key] = texture_sim_score
    texture_sorted_resp_scores_dict = sorted(texture_scores_from_resp_dict.items(), key=lambda x: x[1])
    return texture_sorted_resp_scores_dict[0]


if __name__ == '__main__':
    '''
    load_test_array = np.random.randn(9, 9)
    load_test_array = load_test_array.tolist()
    print(load_test_array)
    load_multi_camera_resp_data('temp', load_test_array)
    my_array = load_test_array[0]
    a = get_texture_from_camera_resp_with_num_block(my_array, 0)
    print(a)
    '''
    # train
    train_mono_img = cv2.imread('test_image/test.bmp')
    camera_response_matrix = np.zeros((9, 9))
    for num_block in range(9):
        camera_response_matrix[num_block] = get_camera_resp_matrix_with_num_block(train_mono_img, num_block)
    #print(camera_response_matrix)
    load_multi_camera_resp_data('face', camera_response_matrix.tolist())

    camera_refl_train = get_refl_wiener_with_resp_list(np.asarray(camera_response_matrix[5]))
    load_multi_textures_data('face', camera_refl_train.tolist())

    # test
    test_mono_img = cv2.imread('test_image/test.bmp')
    test_img_resp = get_camera_resp_matrix_with_num_block(train_mono_img, 3).squeeze()
    result_camera_resp = get_texture_from_camera_resp_with_num_block(test_img_resp, 3)
    print(result_camera_resp)    # 數字越小 越相似    歐氏距離

    test_img_resp = get_camera_resp_matrix_with_num_block(train_mono_img, 3).squeeze()
    camera_refl_test = get_refl_wiener_with_resp_list(np.asarray(test_img_resp))
    result_texture_refl = get_texture_name_with_num_block(camera_refl_test)
    print(result_texture_refl)   # 數字約接近1 越相似  餘弦相似度


