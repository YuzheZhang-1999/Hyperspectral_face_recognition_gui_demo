import cv2
import numpy as np
from skimage import transform as trans
import scipy.io as scio
import get_refl_wiener_pinv.mian_wiener_ref_rec as get_refl
import json
#from sklearn.metrics.pairwise import euclidean_distances


def get_texture(img_array, detect_face_axis):
    paper_face_data_file = '../other_utils/mat_database/paper_face.mat'
    true_face_data_file = '../other_utils/mat_database/true_face.mat'
    screen_face_data_file = '../other_utils/mat_database/screen_face.mat'
    paper_face_data = scio.loadmat(paper_face_data_file)
    true_face_data = scio.loadmat(true_face_data_file)
    screen_face_data = scio.loadmat(screen_face_data_file)

    paper_face_data = paper_face_data['data']
    true_face_data = true_face_data['data']
    screen_face_data = screen_face_data['data']

    mat_shape = true_face_data.shape
    center_axis_x = int(mat_shape[0] / 2)
    center_axis_y = int(mat_shape[1] / 2)

    num_feature_dot = 1
    fea_dot_dis = 5
    true_face_score = 0
    true_face_score_benchmark = 0
    sim_img_array_true = np.zeros(num_feature_dot)
    sim_img_array_screen = np.zeros(num_feature_dot)
    sim_img_array_paper = np.zeros(num_feature_dot)

    #余弦距离
    sim_img_array_true[0] = np.dot(img_array[detect_face_axis[0], detect_face_axis[1], :] , true_face_data[center_axis_x, center_axis_y, :]) \
                            / (np.linalg.norm(img_array[detect_face_axis[0], detect_face_axis[1], :]) * np.linalg.norm(true_face_data[center_axis_x, center_axis_y, :]))

    sim_img_array_screen[0] = np.dot(img_array[detect_face_axis[0], detect_face_axis[1], :] , screen_face_data[center_axis_x, center_axis_y, :]) \
                            / (np.linalg.norm(img_array[detect_face_axis[0], detect_face_axis[1], :]) * np.linalg.norm(screen_face_data[center_axis_x, center_axis_y, :]))

    sim_img_array_paper[0] = np.dot(img_array[detect_face_axis[0], detect_face_axis[1], :] , paper_face_data[center_axis_x, center_axis_y, :]) \
                              / (np.linalg.norm(img_array[detect_face_axis[0], detect_face_axis[1], :]) * np.linalg.norm(paper_face_data[center_axis_x, center_axis_y, :]))

    print('ok')

    for i in range(num_feature_dot):
        print(sim_img_array_screen[i])
        print(sim_img_array_true[i])
        if sim_img_array_screen[i] < sim_img_array_true[i]:
            true_face_score += 1
    
    if true_face_score > true_face_score_benchmark:
        return 'true'
    #elif sim_img_array_paper < sim_img_array_true:
    #    return 'paper'
    else:
        return 'screen'


def load_texture_data(texture_name, texture_refl_list):
    texture_json_file = '../other_utils/json_database/texture_database.json'
    new_data = {texture_name: texture_refl_list}
    try:
        with open(texture_json_file, 'r',  encoding='utf-8') as f:
            old_data = json.load(f)
            load_data = old_data
            load_data.update(new_data)
    except:
        load_data = new_data
    #load_json_data = json.dumps(load_data, indent=4)
    load_json_data = json.dumps(load_data, sort_keys=True, indent=4)
    with open(texture_json_file, 'w', encoding='utf-8') as texture_file:
        texture_file.write(load_json_data)


def get_texture_refl_wiener(img_array):
    train_camera_resp = '../get_refl_wiener_pinv/train_txt_resp_refl/train_camera_resp.txt'
    train_spectral_refl = '../get_refl_wiener_pinv/train_txt_resp_refl/train_spectral_refl.txt'
    resp_train = np.loadtxt(train_camera_resp, dtype=np.float64, delimiter=',')
    refl_train = np.loadtxt(train_spectral_refl, dtype=np.float64, delimiter=',')
    resp_test = get_refl.get_camera_resp_with_image_array(img_array)

    refl_test = get_refl.func_wiener_ref_rec(resp_train, refl_train, resp_test)
    #print('get test refl curve!')
    return refl_test


def get_texture_name(refl_test):
    texture_json_file = '../other_utils/json_database/texture_database.json'
    texture_scores_dict = {}
    with open(texture_json_file, 'r', encoding='utf-8') as texture_json_file:
        texture_database = json.load(texture_json_file)
        for texture_key in texture_database.keys():
            texture_feature = texture_database[texture_key]
            texture_sim_score = np.dot(refl_test, texture_feature) / (np.linalg.norm(refl_test) * np.linalg.norm(texture_feature))
            texture_scores_dict[texture_key] = texture_sim_score
    texture_sorted_scores_dict = sorted(texture_scores_dict.items(), key = lambda x:x[1], reverse = True)
    return texture_sorted_scores_dict[0]


def get_camera_resp_matrix(input_image):
    pic_feature_axis = [[1024,800],[1185,800],[1355,800],[1535,800],[1024,800],[1185,970],[1355,970],[1535,970],[1535,110]]
    camera_resp = np.zeros((1, 9))
    try:
        image_array = input_image[:, :, 0]
    except:
        image_array = input_image
    for i in range(9):
        camera_resp[0, i] = image_array[pic_feature_axis[i][1], pic_feature_axis[i][0]]
    camera_resp = camera_resp/255
    return camera_resp


def load_camera_resp_data(camera_resp_object_name, camera_resp_list):
    camera_resp_json_file = '../other_utils/json_database/camera_resp_database.json'
    new_data = {camera_resp_object_name: camera_resp_list}
    try:
        with open(camera_resp_json_file, 'r',  encoding='utf-8') as f:
            old_data = json.load(f)
            load_data = old_data
            load_data.update(new_data)
    except:
        load_data = new_data
    #load_json_data = json.dumps(load_data, indent=4)
    load_json_data = json.dumps(load_data, sort_keys=True, indent=4)
    with open(camera_resp_json_file, 'w', encoding='utf-8') as f:
        f.write(load_json_data)


def get_texture_from_camera_resp(resp_test):
    camera_resp_json_file = '../other_utils/json_database/camera_resp_database.json'
    texture_scores_from_resp_dict = {}
    with open(camera_resp_json_file, 'r', encoding='utf-8') as f:
        camera_resp_database = json.load(f)
        for camera_resp_key in camera_resp_database.keys():
            camera_resp_list = camera_resp_database[camera_resp_key]
            texture_sim_score = np.linalg.norm(resp_test - camera_resp_list)
            texture_scores_from_resp_dict[camera_resp_key] = texture_sim_score
    texture_sorted_resp_scores_dict = sorted(texture_scores_from_resp_dict.items(), key = lambda x:x[1])
    return texture_sorted_resp_scores_dict[0]




if __name__ == '__main__':
    load_camera_resp_data('aagfdsce', [123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123])
    my_array = np.asarray([123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123])
    a = get_texture_from_camera_resp(my_array)
    print(a)
