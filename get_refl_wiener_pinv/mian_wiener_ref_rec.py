import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
'''
将matlab中的函数func_wiener_ref_rec 用python进行实现，其中需要重写的函数有
func_pinv_sens   计算各通道的系统相应函数
func_my_pinv     在上述函数中用到了该函数，是根据svd原理，自己完成计算伪逆的功能，以便控制
repmat matlab自带的函数，repmat(A,m,n)将矩阵A复制m*n块
diag  matlab自带函数 
pinv matlab自带的求伪逆的函数
'''


'''
b = np.linalg.pinv(a)     #求a的伪拟 b
'''


def func_my_pinv(A, tol):
    U, S, V = np.linalg.svd(A)          # A = USV
    V = V.T
    B = np.zeros(A.shape)
    np.fill_diagonal(B, S)              # 用S填充B的对角线元素，因为python中奇异值分解出的S不是完整的矩阵
    S = B
    sc = np.diag(S)
    ind = (np.where(sc > sc[0]*tol))[0]
    sc_inv = np.zeros((S.T).shape)
    for i in range(len(ind)):
        sc_inv[i, i] = 1 / sc[i]
    Ainv = (V.dot(sc_inv)).dot(U.T)      #V sc_inv U' 矩阵相乘 得到伪逆
    return Ainv


def func_pinv_sens(resp, refl, tol):        # refl 31*N    resp 9*N
    N, num = refl.shape
    channel = resp.shape[0]
    N1 = N+1

    sens1 = np.zeros((N1, channel))

    C = np.ones((num, N1))
    C[:, 0:N] = refl.T
    for i in range(channel):
        d = resp[i,:]
        d = d.T
        x = func_my_pinv(C, tol).dot(d)
        sens1[:, i] = x
    resp_pred = (C.dot(sens1)).T
    sens = sens1[0:N, :].T
    const_bias = sens1[-1, :].T
    return sens, const_bias, resp_pred


def func_wiener_ref_rec(train_X, train_Y, test_X):
    resp_train = train_X.T
    refl_train = train_Y.T
    resp_test = test_X.T
    # 计算各通道的系统响应函数
    (M, bias, resp_pred) = func_pinv_sens(resp_train, refl_train, 0.001)
    #估计噪声水平
    bias_T = bias.T
    bias_mat = (np.tile(bias_T, resp_train.shape[1]).reshape(resp_train.shape[1],-1)).T    # 将矩阵沿着2维方向复制resp_train.shape[1]个

    resp_train = resp_train -  bias_mat             ## 可能存在问题

    resp_noise = resp_train - M.dot(refl_train)
    Kn = (resp_noise.dot(resp_noise.T))/resp_noise.shape[1]
    Kn = np.diag(np.diag(Kn))
    K_train = (refl_train.dot(refl_train.T))/refl_train.shape[1]

    #维纳估计
    W = (K_train.dot(M.T)).dot(np.linalg.pinv((M.dot(K_train)).dot(M.T)+Kn))
    refl_test_p = W.dot(resp_test - bias)
    return refl_test_p


def get_camera_resp(image_path_dir):
    pic_num = len(os.listdir(image_path_dir))
    pic_feature_axis = [[1024, 800], [1185, 800], [1355, 800], [1535, 800],[1024, 800], [1185, 970], [1355, 970],[1535, 970],[1535, 110]]
    camera_resp = np.zeros((pic_num, 9))
    for root_dir, sub_dir, files in os.walk(image_path_dir):
        for pic_count, file in enumerate(files):
            file_name = os.path.join(root_dir, file)
            print(file_name)
            image_array = cv2.imread(file_name)
            for i in range(9):
                camera_resp[pic_count, i] = image_array[pic_feature_axis[i][1], pic_feature_axis[i][0], 0]
    camera_resp = (camera_resp / 255).squeeze()
    return camera_resp


def get_camera_resp_with_image_array(image_array):
    pic_feature_axis = [[1024, 800], [1185, 800], [1355, 800], [1535, 800],[1024, 800], [1185, 970], [1355, 970],[1535, 970],[1535, 110]]
    camera_resp = np.zeros((1, 9))
    for i in range(9):
        try:
            camera_resp[0, i] = image_array[pic_feature_axis[i][1], pic_feature_axis[i][0]]
        except:
            camera_resp[0, i] = image_array[pic_feature_axis[i][1], pic_feature_axis[i][0], 0]     #if the input is 3-dimension
    camera_resp = (camera_resp / 255).squeeze()
    #print(camera_resp)
    return camera_resp


if __name__ == '__main__':
    #A = np.array([[1, 2, 3], [4, 5, 6]])
    #print(func_my_pinv(A, 0.001))    #测试自己写的求伪逆的函数

    #resp_train = np.loadtxt('./test_data/image_feature_resp.txt', dtype=np.float64, delimiter=',')
    #refl_train = np.loadtxt('./test_data/spectral_curve_data.txt', dtype=np.float64, delimiter=',')
    #resp_train = resp_train.T
    #refl_train = refl_train.T
    #(sens, constbias, resp_pred) = func_pinv_sens(resp_train, refl_train, 0.001)
    #print(constbias)                 # 测试自己写的维纳估计的函数

    test_image_path = './test_image_one_pic'
    train_camera_resp = './train_txt_resp_refl/train_camera_resp.txt'
    train_spectral_refl = './train_txt_resp_refl/train_spectral_refl.txt'
    resp_test = get_camera_resp(test_image_path)
    resp_train = np.loadtxt(train_camera_resp, dtype=np.float64, delimiter=',')
    refl_train = np.loadtxt(train_spectral_refl, dtype=np.float64, delimiter=',')

    refl_test = func_wiener_ref_rec(resp_train, refl_train, resp_test)
    print('get test refl curve!')
    axis_x = np.linspace(420, 720, num=31)
    plt.title('Spectral curve')
    plt.plot(axis_x, refl_test)
    plt.pause(0.1)


