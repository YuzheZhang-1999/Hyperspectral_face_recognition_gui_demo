from __future__ import print_function
import sys

sys.path.append('../')
sys.path.append('../jai_package')
sys.path.append('../retinaface_torch')
sys.path.append('../retinaface_torch/weights')
sys.path.append('../get_refl_wiener_pinv')

import cv2
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
import threading

from other_utils import face_align, detect_texture, get_overlay_diagram

from retinaface_torch.data import cfg_mnet, cfg_re50
from retinaface_torch.layers.functions.prior_box import PriorBox
from retinaface_torch.utils.nms.py_cpu_nms import py_cpu_nms
from retinaface_torch.utils.box_utils import decode, decode_landm
from retinaface_torch.models.retinaface import RetinaFace

from arcface_torch.backbones import get_model

'''
from unet_rec_img.model import MultiscaleNet_9
from unet_rec_img import dataset_test as dataset
from unet_rec_img.HSI2RGB import HSI2RGB
from torch.utils.data import DataLoader
'''
from UNet_N_31.model import UNet
from UNet_N_31 import dataset_old as dataset
from UNet_N_31.HSI2RGB import HSI2RGB
from torch.utils.data import DataLoader

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from system_gui import *
from jai_package import jai_camera_package


# retinaface
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_retinaface():
    pretrained_path = '../retinaface_torch/weights/mobilenet0.25_Final.pth'
    torch.set_grad_enabled(False)
    device = torch.cuda.current_device()
    retinaface_net = RetinaFace(cfg_mnet, phase='test')
    retinaface_net = load_model(retinaface_net, pretrained_path, False)
    retinaface_net.eval()
    print('Finished loading retinaface model!')
    cudnn.benchmark = True
    device = torch.device("cuda")
    retinaface_net = retinaface_net.to(device)
    return retinaface_net


def retinaface_detect(net, image, cfg=cfg_mnet):
    confidence_threshold = 0.6
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    vis_thres = 0.6
    img_raw = image
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    device = torch.device("cuda")
    img = img.to(device)
    scale = scale.to(device)
    loc, conf, landms = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    dets = np.concatenate((dets, landms), axis=1)

    return dets


def load_arcface():
    pretrained_path = '../arcface_torch/ms1mv3_arcface_r18_fp16/backbone.pth'
    device = torch.device("cuda:0")
    net = get_model('r18', fp16=True)
    net.load_state_dict(torch.load(pretrained_path))
    net = net.to(device)
    net.eval()
    print('Finished loading arcface model!')
    return net


def arcface_getfeature(net, image):
    device = torch.device("cuda:0")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    img = img.to(device)
    feat = net(img).cpu().numpy()[0]
    norm = np.sqrt(np.sum(feat * feat) + 0.00001)
    feat /= norm
    return feat


def load_MSnet():
    device = torch.device("cuda:0")
    MSnet = UNet.MultiscaleNet().to(device)
    checkpoint_path = '../UNet_N_31/checkpoint/checkpoint_0720_2221/checkpoint_9_599.pt'
    MSnet.load_state_dict(torch.load(checkpoint_path, map_location=device))
    MSnet.eval()
    print('Finished loading unet model!')
    return MSnet


def msnet_get_recimage(net, image):
    device = torch.device("cuda:0")
    image = image[int(image.shape[0] / 2) - 1023:int(image.shape[0] / 2) + 1024,
            int(image.shape[1] / 2) - 1023:int(image.shape[1] / 2) + 1024, 0].astype('float32')
    image = image[0:image.shape[0]:8, 0:image.shape[1]:8]

    image_tensor = ((torch.from_numpy(image)).unsqueeze(0).unsqueeze(0)).to(device)
    # print(image_tensor.shape)
    hsi_image = net(image_tensor)
    hsi_image = hsi_image.squeeze().permute(1, 2, 0)
    # print(hsi_image)
    # print(hsi_image.shape)
    return hsi_image


def my_resize_256(image):
    image = image[int(image.shape[0] / 2) - 1023:int(image.shape[0] / 2) + 1024,
            int(image.shape[1] / 2) - 1023:int(image.shape[1] / 2) + 1024, :].astype('float32')
    image = image[0:image.shape[0]:8, 0:image.shape[1]:8, :]
    return image


# other function 绘制9个点的光谱曲线
def show_spectral_curve(hyperspectral_mat, spectral_curve_wiener):
    hsi_rebuilt_shape = hyperspectral_mat.shape
    '''
    fea_dot_dis = 5
    center_axis_x = int(hsi_rebuilt_shape[0] / 2)
    center_axis_y = int(hsi_rebuilt_shape[1] / 2)
    hyper_spec_line_1 = np.squeeze(hyperspectral_mat[center_axis_x - fea_dot_dis, center_axis_y - fea_dot_dis, :])
    hyper_spec_line_2 = np.squeeze(hyperspectral_mat[center_axis_x, center_axis_y - fea_dot_dis, :])
    hyper_spec_line_3 = np.squeeze(hyperspectral_mat[center_axis_x + fea_dot_dis, center_axis_y - fea_dot_dis, :])
    hyper_spec_line_4 = np.squeeze(hyperspectral_mat[center_axis_x - fea_dot_dis, center_axis_y, :])
    hyper_spec_line_5 = np.squeeze(hyperspectral_mat[center_axis_x, center_axis_y, :])
    hyper_spec_line_6 = np.squeeze(hyperspectral_mat[center_axis_x + fea_dot_dis, center_axis_y, :])
    hyper_spec_line_7 = np.squeeze(hyperspectral_mat[center_axis_x - fea_dot_dis, center_axis_y + fea_dot_dis, :])
    hyper_spec_line_8 = np.squeeze(hyperspectral_mat[center_axis_x, fea_dot_dis + fea_dot_dis, :])
    hyper_spec_line_9 = np.squeeze(hyperspectral_mat[center_axis_x + 20, center_axis_y + fea_dot_dis, :])
    axis_x = np.linspace(420, 720, num=31)
    plt.clf()
    plt.title('Spectral curve')
    plt.subplot(3, 3, 1)
    plt.plot(axis_x, hyper_spec_line_1)
    plt.subplot(3, 3, 2)
    plt.plot(axis_x, hyper_spec_line_2)
    plt.subplot(3, 3, 3)
    plt.plot(axis_x, hyper_spec_line_3)
    plt.subplot(3, 3, 4)
    plt.plot(axis_x, hyper_spec_line_4)
    plt.subplot(3, 3, 5)
    plt.plot(axis_x, hyper_spec_line_5)
    plt.subplot(3, 3, 6)
    plt.plot(axis_x, hyper_spec_line_6)
    plt.subplot(3, 3, 7)
    plt.plot(axis_x, hyper_spec_line_7)
    plt.subplot(3, 3, 8)
    plt.plot(axis_x, hyper_spec_line_8)
    plt.subplot(3, 3, 9)
    plt.plot(axis_x, hyper_spec_line_9)
    '''
    # plt.figure(2)
    axis_x = np.linspace(420, 720, num=31)
    plt.clf()
    plt.title('Spectral curve by wiener')
    plt.plot(axis_x, spectral_curve_wiener)  # 显示wiener预测的光谱曲线
    plt.pause(0.001)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        # 设置四个用于显示信息的button的背景色 and 去掉button的边框

        self.label_text_name_right.setStyleSheet("background-color: white")
        self.label_text_conf_right.setStyleSheet("background-color: white")
        self.label_text_name_right.setStyleSheet("border:none;")
        self.label_text_conf_right.setStyleSheet("border:none;")

        # 用于描述rgb相机和单色相机对于人脸采集后识别的结果 置信度 以及 材质置信度
        self.rgb_face_result = 0
        self.mono_face_result = 0
        self.mono_face_texture_name = 'face'
        self.mono_face_texture_score = 0
        self.mono_face_texture_result = (self.mono_face_texture_name, self.mono_face_texture_score)
        self.mono_face_center_axis = [1000, 1000]

        self.texture_name_from_camera_resp = 'face'
        self.texture_score_from_camera_resp = 0
        self.texture_result_from_camera_resp = (self.texture_name_from_camera_resp, self.texture_score_from_camera_resp)

        self.texture_name_from_joint_forecast = ''

        # 用于存储重建的31通道的mat图像
        self.hsi_rebuilt_save_mat = 0
        self.rebuilt_image_scale = 2
        # self.rebuild_base_image = my_resize_256(cv2.imread('./base.bmp'))
        self.rebuild_base_image = 1
        self.base_image_scale = 1  # *5 liang du

        # 决定是否显示光谱图像
        self.show_spectral_curve_flag = 0

        # 在gui上显示的光谱曲线的图片
        self.spectral_curve_image = np.ones((500, 500, 3)) * 255

        # 用于保存数据库人脸数据的路径 以及 数据库人脸的特征值
        self.face_rgb_database_filepath = './Database_rgb/temp.jpg'
        self.face_mono_database_filepath = './Database_mono/temp.bmp'
        self.face_rgb_database_feat = 0
        self.face_mono_database_feat = 0
        self.texture_name = 'temp'
        self.spectral_curve_wiener = np.zeros((1, 31))  # 用于保存经过wiener估计得到的光谱曲线数据

        # 用于表示人脸识别是否开始的 flag
        self.face_rgb_recognition_flag = 0
        self.face_mono_recognition_flag = 0

        # 用于获取 jai相机图像 以及 描述获取状态的变量
        self.jai_go5000C_camera = 0
        self.jai_go5000C_image = np.zeros((2048, 2560, 1), np.uint8)
        self.jai_go5000C_open_state = 0

        self.jai_go5100M_camera = 0
        self.jai_go5100M_image = np.zeros((2056, 2464, 1), np.uint8)
        self.jai_go5100M_rebuild_image = np.zeros((2056, 2464, 3), np.uint8)
        self.jai_go5100M_open_state = 0

        # 定时器 用于定时获取每帧图像
        self.timer_camera_c = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer_camera_m = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer_camera_c.timeout.connect(self.show_camera_c)
        self.timer_camera_m.timeout.connect(self.show_camera_m)

        # 加载 retinaface
        self.retinaface_net = load_retinaface()

        # 加载 arcface
        self.arcface_net = load_arcface()

        # 加载 unet
        self.msnet_unet = load_MSnet()

        # ************* plt
        plt.ion()

    def open_camera_1(self):
        if not self.timer_camera_c.isActive():  # 若定时器未启动
            try:
                self.jai_go5000C_camera = jai_camera_package.JaiCamera(0)
                self.jai_go5000C_open_state = self.jai_go5000C_camera.open_camera()
                self.jai_go5000C_image = self.jai_go5000C_camera.get_raw_image()
            except:
                self.jai_go5000C_open_state = 0
                msg = QMessageBox.warning(self, 'warning', "请检查彩色相机是否连接正确", buttons=QMessageBox.Ok)
            else:
                self.timer_camera_c.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.pushButton_open_camera_1.setText('关闭相机1')
        else:
            self.timer_camera_c.stop()  # 关闭定时器
            if self.jai_go5000C_open_state:
                self.jai_go5000C_camera.close_camera()
            self.jai_go5000C_open_state = 0
            self.face_rgb_recognition_flag = 0
            self.label_show_rgb_image.clear()  # 清空视频显示区域
            self.pushButton_open_camera_1.setText('打开相机1')
            self.label_show_rgb_image.setPixmap(QtGui.QPixmap("ICON/cap_face_icon.jpg"))

    def show_camera_c(self):
        go5000_image = cv2.cvtColor(self.jai_go5000C_image, cv2.COLOR_BAYER_GR2BGR)
        go5000_image = cv2.resize(go5000_image, (0, 0), fx=0.1, fy=0.1)
        if self.jai_go5000C_open_state and self.face_rgb_recognition_flag:
            try:
                bbox_pts = retinaface_detect(self.retinaface_net, go5000_image)[0]
                face_current_bbox = bbox_pts[0:4]
                face_current_pts5 = np.zeros((5, 2))
                for i in range(5):
                    for j in range(2):
                        face_current_pts5[i, j] = bbox_pts[5 + 2 * i + j]
                # 经过retinaface得到人脸关键点后，经过人脸对齐算法 得到（112，112）的人脸图片
                align_face_img = face_align.norm_crop(go5000_image, face_current_pts5)
                showImage = QtGui.QImage(align_face_img.data, align_face_img.shape[1], align_face_img.shape[0],
                                         align_face_img.shape[1] * 3, QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_show_rgb_align_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
                cv2.rectangle(go5000_image, (int(face_current_bbox[0]), int(face_current_bbox[1])),
                              (int(face_current_bbox[2]), int(face_current_bbox[3])), (0, 255, 0), 1)
                # 获取人脸特征向量 并 与数据库人脸进行比对 得到置信度
                face_rgb_current_feat = arcface_getfeature(self.arcface_net, align_face_img)
                self.rgb_face_result = np.dot(self.face_rgb_database_feat, face_rgb_current_feat)
                self.rgb_face_result = round(self.rgb_face_result, 3)
                self.rgb_face_result = self.rgb_face_result * 2.5 if self.rgb_face_result * 2.5 < 1 else 1
                self.label_text_name_left.setText(
                    '名称_材质:' + self.face_rgb_database_filepath.split('/')[-1].split('.')[0] + ' ' + 'face')
                self.label_text_conf_left.setText('置信度:' + str(round(self.rgb_face_result, 3)))
            except:
                pass
        showImage = QtGui.QImage(go5000_image.data, go5000_image.shape[1],
                                 go5000_image.shape[0], go5000_image.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_rgb_image.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def select_face_1(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选择比对人脸", os.getcwd(),
                                                                   "JPG Files(*.jpg);;PNG Files(*.png)")
        self.face_rgb_database_filepath = fileName
        try:
            show = cv2.imread(self.face_rgb_database_filepath)
            try:
                show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
            except:
                pass
            self.label_text_name_left.setText(
                "名称_材质" + '\n' + self.face_rgb_database_filepath.split('/')[-1].split('.')[0] + '_')
            # showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3,QtGui.QImage.Format_RGB888)
            # self.label_choice_face_left.setPixmap(QtGui.QPixmap.fromImage(showImage))

        except:
            print('Did not choice file')

    def collect_face_1(self):
        if self.jai_go5000C_open_state:
            show = cv2.cvtColor(self.jai_go5000C_image, cv2.COLOR_BAYER_GR2RGB)
            cv2.imwrite('./Database_rgb/wyg.jpg', show)
        else:
            msg = QMessageBox.warning(self, 'warning', "请先打开相机", buttons=QMessageBox.Ok)

    def start_recognition_1(self):
        if self.jai_go5000C_open_state:
            print('Using insightface')
            database_img = cv2.imread(self.face_rgb_database_filepath)
            bbox_pts = retinaface_detect(self.retinaface_net, database_img)[0]
            database_face_pts5 = np.zeros((5, 2))
            for i in range(5):
                for j in range(2):
                    database_face_pts5[i, j] = bbox_pts[5 + 2 * i + j]
            align_face_img = face_align.norm_crop(database_img, database_face_pts5)
            self.face_rgb_database_feat = arcface_getfeature(self.arcface_net, align_face_img)
            self.face_rgb_recognition_flag = 1
        else:
            msg = QMessageBox.warning(self, 'warning', "请先打开相机", buttons=QMessageBox.Ok)

    def select_font(self):
        font, ok = QtWidgets.QFontDialog.getFont()
        if ok:
            self.label_text_conf_right.setFont(font)
            self.label_text_name_right.setFont(font)

    def init_system(self):
        self.timer_camera_m.stop()  # 关闭定时器
        if self.jai_go5100M_open_state:
            self.jai_go5100M_camera.close_camera()
        self.jai_go5100M_open_state = 0
        self.face_mono_recognition_flag = 0
        self.label_origin_image_icon.clear()  # 清空视频显示区域
        self.pushButton_open_camera_2.setText('打开相机2')
        self.label_text_conf_right.setText("置信度")
        self.label_text_name_right.setText("名称_材质")
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(20)
        self.label_text_conf_right.setFont(font)
        self.label_text_name_right.setFont(font)
        self.label_rebuild_hyper_image.setPixmap(QtGui.QPixmap("ICON/label_rebuild_hyperimage.jpg"))
        self.label_origin_image_icon.setPixmap(QtGui.QPixmap("ICON/label_origin_image_icon.jpg"))
        self.label_show_spectral_curve.setPixmap(QtGui.QPixmap("ICON/label_spectral_curve.jpg"))
        print('Init system finished!')

    def system_exit(self):
        if self.jai_go5100M_open_state:
            self.jai_go5100M_camera.close_camera()
        self.close()

    def open_camera_2(self):
        if not self.timer_camera_m.isActive():  # 若定时器未启动
            try:
                self.jai_go5100M_camera = jai_camera_package.JaiCamera(1)
                self.jai_go5100M_open_state = self.jai_go5100M_camera.open_camera()
                self.jai_go5100M_image = self.jai_go5100M_camera.get_raw_image()
            except:
                self.jai_go5100M_open_state = 0
                msg = QMessageBox.warning(self, 'warning', "请检查单色相机是否连接正确", buttons=QMessageBox.Ok)
            else:
                self.timer_camera_m.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.pushButton_open_camera_2.setText('关闭相机2')
        else:
            self.timer_camera_m.stop()  # 关闭定时器
            if self.jai_go5100M_open_state:
                self.jai_go5100M_camera.close_camera()
            self.jai_go5100M_open_state = 0
            self.face_mono_recognition_flag = 0
            self.label_origin_image_icon.clear()  # 清空视频显示区域
            self.pushButton_open_camera_2.setText('打开相机2')
            self.label_origin_image_icon.setPixmap(QtGui.QPixmap("ICON/label_origin_image_icon.jpg"))

    def show_camera_m(self):
        go5100_image = self.jai_go5100M_image
        # go5100_image = cv2.resize(go5100_image, (0, 0), fx=0.1, fy=0.1)
        if self.jai_go5100M_open_state and self.face_mono_recognition_flag:
            try:
                # 先通过 unet 重建rgb图像  并显示重建的图像
                go5100_rebuilt_image = msnet_get_recimage(self.msnet_unet, go5100_image)
                hsi_rebuilt_save_data = go5100_rebuilt_image
                hsi_rebuilt_save_data = hsi_rebuilt_save_data.cpu().numpy()
                self.hsi_rebuilt_save_mat = hsi_rebuilt_save_data

                # 通过wiener估计的方法得到预测的光谱曲线
                go5100_image = go5100_image.reshape(2056, 2464)
                self.spectral_curve_wiener = detect_texture.get_texture_refl_wiener(go5100_image)

                '''
                # draw the spectrual line #
                if self.show_spectral_curve_flag:
                    show_spectral_curve(self.hsi_rebuilt_save_mat, self.spectral_curve_wiener)
                '''
                ######################################
                # 根据网络重建的高光谱图得到中心点的光谱曲线图，并在gui中显示
                # spectral_curve_image_array = show_spectral_curve_on_gui((go5100_rebuilt_image[int(go5100_rebuilt_image.shape[0]/2), int(go5100_rebuilt_image.shape[1]/2), :]).tolist())
                if self.show_spectral_curve_flag:
                    self.show_spectral_curve_on_gui()
                    spectral_curve_image_array = cv2.resize(self.spectral_curve_image, (500, 500))
                    showImage = QtGui.QImage(spectral_curve_image_array.data, spectral_curve_image_array.shape[1],
                                             spectral_curve_image_array.shape[0],
                                             spectral_curve_image_array.shape[1] * 3,
                                             QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                    self.label_show_spectral_curve.setPixmap(QtGui.QPixmap.fromImage(showImage))

                #######################################
                go5100_rebuilt_image = self.rebuilt_image_scale * (HSI2RGB(go5100_rebuilt_image))

                go5100_rebuilt_image = self.base_image_scale * (
                    np.divide(go5100_rebuilt_image, self.rebuild_base_image))
                go5100_rebuilt_image = (go5100_rebuilt_image.numpy()).astype(np.uint8)

                go5100_rebuilt_overlay_image = get_overlay_diagram.get_overlay_image_from_hyperspectral(
                    (self.hsi_rebuilt_save_mat) * 20)
                # resize重建的光谱曲线，因为生成的是256*256，我们把他resize成500*500
                go5100_rebuilt_overlay_image = cv2.resize(go5100_rebuilt_overlay_image, (500, 500))
                showImage = QtGui.QImage(go5100_rebuilt_overlay_image.data, go5100_rebuilt_overlay_image.shape[1],
                                         go5100_rebuilt_overlay_image.shape[0],
                                         go5100_rebuilt_overlay_image.shape[1] * 3,
                                         QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_rebuild_hyper_image.setPixmap(QtGui.QPixmap.fromImage(showImage))

                # 判断当前任务是识别人脸还是识别普通物品
                if self.combo_select_face_object.currentText() == 'face':
                    # 再进行识别
                    bbox_pts = retinaface_detect(self.retinaface_net, go5100_rebuilt_image)[0]
                    face_current_bbox = bbox_pts[0:4]
                    face_current_pts5 = np.zeros((5, 2))
                    for i in range(5):
                        for j in range(2):
                            face_current_pts5[i, j] = bbox_pts[5 + 2 * i + j]
                    self.mono_face_center_axis[0] = int((face_current_bbox[0] + face_current_bbox[2]) / 2)
                    self.mono_face_center_axis[1] = int((face_current_bbox[1] + face_current_bbox[3]) / 2)

                    # get face texture information
                    # 这个是使用网络重建的高光谱 进行材质判别
                    # self.mono_face_texture_name = detect_texture.get_texture(hsi_rebuilt_save_data,self.mono_face_center_axis)

                    # 这个是使用维纳估计恢复的一条光谱曲线进行判别
                    self.mono_face_texture_name = detect_texture.get_texture_name(self.spectral_curve_wiener)[0]
                    current_camera_resp_list = (detect_texture.get_camera_resp_matrix(self.jai_go5100M_image))
                    self.texture_name_from_camera_resp = \
                    detect_texture.get_texture_from_camera_resp(current_camera_resp_list)[0]
                    if self.mono_face_texture_name == self.texture_name_from_camera_resp == 'face':
                        self.texture_name_from_joint_forecast = 'face'
                    elif self.mono_face_texture_name == self.texture_name_from_camera_resp == 'screen':
                        self.texture_name_from_joint_forecast = 'screen'
                    self.texture_name = self.face_mono_database_filepath.split('/')[-1].split('.')[
                                            0] + ' ' + self.texture_name_from_joint_forecast

                    # if self.mono_face_texture_name == 'true':
                    #    self.texture_name = self.face_mono_database_filepath.split('/')[-1].split('.')[0]
                    #    # self.label_text_name_right.setText('姓名:' + self.face_mono_database_filepath.split('/')[-1].split('.')[0])
                    # elif self.mono_face_texture_name == 'screen':
                    #    self.texture_name = 'fake_screen'

                    # 经过retinaface得到人脸关键点后，经过人脸对齐算法 得到（112，112）的人脸图片

                    align_face_img = face_align.norm_crop(go5100_rebuilt_image, face_current_pts5)
                    cv2.rectangle(go5100_rebuilt_image, (int(face_current_bbox[0]), int(face_current_bbox[1])),
                                  (int(face_current_bbox[2]), int(face_current_bbox[3])), (0, 255, 0), 1)
                    # 获取人脸特征向量 并 与数据库人脸进行比对 得到置信度
                    face_mono_current_feat = arcface_getfeature(self.arcface_net, align_face_img)
                    self.mono_face_result = np.dot(self.face_mono_database_feat, face_mono_current_feat)
                    self.mono_face_result = round(self.mono_face_result, 3)
                    self.label_text_name_right.setText('名称_材质:' + self.texture_name)
                    if self.texture_name_from_joint_forecast == 'face':
                        self.mono_face_result = self.mono_face_result * 1.5 if self.mono_face_result * 1.5 < 1 else 1
                    else:
                        self.mono_face_result = self.mono_face_result / 2 if self.mono_face_result / 2 > 0.1 else 0.1
                    self.mono_face_result = round(self.mono_face_result, 3)
                    self.label_text_conf_right.setText('置信度:' + str(self.mono_face_result))
                elif self.combo_select_face_object.currentText() == 'object':
                    print('object')
                    '''
                    # 再进行识别
                    bbox_pts = retinaface_detect(self.retinaface_net, go5100_rebuilt_image)[0]
                    face_current_bbox = bbox_pts[0:4]
                    face_current_pts5 = np.zeros((5, 2))
                    for i in range(5):
                        for j in range(2):
                            face_current_pts5[i, j] = bbox_pts[5 + 2 * i + j]
                    self.mono_face_center_axis[0] = int((face_current_bbox[0] + face_current_bbox[2]) / 2)
                    self.mono_face_center_axis[1] = int((face_current_bbox[1] + face_current_bbox[3]) / 2)

                    # get face texture information
                    # 这个是使用网络重建的高光谱 进行材质判别
                    #self.mono_face_texture_name = detect_texture.get_texture(hsi_rebuilt_save_data,self.mono_face_center_axis)
                    '''
                    # 这个是使用维纳估计恢复的一条光谱曲线进行判别
                    self.mono_face_texture_result = detect_texture.get_texture_name(self.spectral_curve_wiener)
                    self.mono_face_texture_name = self.mono_face_texture_result[0]
                    self.mono_face_texture_score = self.mono_face_texture_result[1]

                    current_camera_resp_list = (detect_texture.get_camera_resp_matrix(self.jai_go5100M_image))
                    self.texture_result_from_camera_resp = detect_texture.get_texture_from_camera_resp(
                        current_camera_resp_list)
                    self.texture_name_from_camera_resp = self.texture_result_from_camera_resp[0]
                    self.texture_score_from_camera_resp = self.texture_result_from_camera_resp[1]

                    if self.mono_face_texture_name == self.texture_name_from_camera_resp == 'leaf':
                        self.texture_name_from_joint_forecast = 'leaf'
                    elif self.mono_face_texture_name == self.texture_name_from_camera_resp == 'tank':
                        self.texture_name_from_joint_forecast = 'tank'
                    else:
                        self.texture_name_from_joint_forecast = ' '
                    self.texture_name = self.texture_name_from_joint_forecast
                    # print(self.texture_name)

                    if self.texture_name_from_joint_forecast != ' ':
                        if self.texture_name_from_joint_forecast == 'tank':
                            self.mono_face_result = self.mono_face_texture_score
                            self.mono_face_result = self.mono_face_result * 1.5 if self.mono_face_result * 1.5 < 1 else 1
                        elif self.texture_name_from_joint_forecast == 'leaf':
                            self.mono_face_result = 1 - self.mono_face_texture_score
                            self.mono_face_result = self.mono_face_result / 2 if self.mono_face_result / 2 > 0.1 else 0.1
                        self.mono_face_result = round(self.mono_face_result, 3)
                        self.label_text_name_right.setText('名称_材质:' + self.texture_name)
                        self.label_text_conf_right.setText('置信度:' + str(self.mono_face_result))
                    else:
                        self.label_text_name_right.setText('名称_材质:')
                        self.label_text_conf_right.setText('置信度:')
            except:
                pass
        go5100_image = go5100_image.reshape(2056, 2464, 1)
        showImage = QtGui.QImage(go5100_image.data, go5100_image.shape[1],
                                 go5100_image.shape[0], go5100_image.shape[1],
                                 QtGui.QImage.Format_Grayscale8)  # 把读取到的视频数据变成QImage形式
        self.label_origin_image_icon.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def select_face_2(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选择比对人脸", os.getcwd(),
                                                                   "BMP Files(*.bmp)")
        self.face_mono_database_filepath = fileName
        try:
            show = cv2.imread(self.face_mono_database_filepath)
        except:
            print('Did not choice file')

    def collect_face_2(self):
        if self.jai_go5100M_open_state:
            show = self.jai_go5100M_image
            cv2.imwrite('./Database_mono/zyz.bmp', show)
        else:
            msg = QMessageBox.warning(self, 'warning', "请先打开相机", buttons=QMessageBox.Ok)

    def choice_show_spectral_curve(self):
        if self.face_mono_recognition_flag == self.jai_go5100M_open_state == 1:
            if self.show_spectral_curve_flag:
                self.show_spectral_curve_flag = 0
                self.label_show_spectral_curve.setPixmap(QtGui.QPixmap("ICON/label_spectral_curve.jpg"))
                self.pushButton_show_spectral_curve.setText('显示光谱曲线')
            else:
                self.show_spectral_curve_flag = 1
                self.pushButton_show_spectral_curve.setText('关闭光谱曲线')
        else:
            msg = QMessageBox.warning(self, 'warning', "请打开相机并开始识别", buttons=QMessageBox.Ok)

    def collect_hyperspectral_image(self):
        if self.jai_go5100M_open_state:
            if self.face_mono_recognition_flag:
                sio.savemat('./hyperspectral_image.mat', {'data': self.hsi_rebuilt_save_mat})
                if self.line_edit_get_texture_text.text() != None:
                    texture_name = self.line_edit_get_texture_text.text()
                    print('adding texture!')
                    detect_texture.load_texture_data(texture_name, (self.spectral_curve_wiener).tolist())
                    camera_resp_list = (detect_texture.get_camera_resp_matrix(self.jai_go5100M_image)).tolist()
                    detect_texture.load_camera_resp_data(texture_name, camera_resp_list)
            else:
                msg = QMessageBox.warning(self, 'warning', "请先开始识别", buttons=QMessageBox.Ok)
        else:
            msg = QMessageBox.warning(self, 'warning', "请先打开相机", buttons=QMessageBox.Ok)

    def start_recognition_2(self):
        if self.jai_go5100M_open_state:
            print('Starting recognition......')
            database_img = cv2.imread(self.face_mono_database_filepath)
            # 先通过 unet 重建rgb图像
            go5100_rebuilt_image = msnet_get_recimage(self.msnet_unet, database_img)
            go5100_rebuilt_image = self.rebuilt_image_scale * (HSI2RGB(go5100_rebuilt_image))
            go5100_rebuilt_image = self.base_image_scale * (np.divide(go5100_rebuilt_image, self.rebuild_base_image))
            go5100_rebuilt_image = (go5100_rebuilt_image.numpy()).astype(np.uint8)
            # 再进行识别

            bbox_pts = retinaface_detect(self.retinaface_net, go5100_rebuilt_image)[0]
            database_face_pts5 = np.zeros((5, 2))
            for i in range(5):
                for j in range(2):
                    database_face_pts5[i, j] = bbox_pts[5 + 2 * i + j]
            align_face_img = face_align.norm_crop(go5100_rebuilt_image, database_face_pts5)
            self.face_mono_database_feat = arcface_getfeature(self.arcface_net, align_face_img)

            self.face_mono_recognition_flag = 1
        else:
            msg = QMessageBox.warning(self, 'warning', "请先打开相机", buttons=QMessageBox.Ok)

    def show_spectral_curve_on_gui(self):
        go5100_rebuilt_image = self.hsi_rebuilt_save_mat
        spectral_curve_list = (
            go5100_rebuilt_image[int(go5100_rebuilt_image.shape[0] / 2), int(go5100_rebuilt_image.shape[1] / 2),
            :]).tolist()
        axis_x = np.linspace(420, 720, num=31)
        plt.clf()
        plt.title('Spectral curve')
        plt.plot(axis_x, spectral_curve_list)  # 显示wiener预测的光谱曲线
        canvas = FigureCanvasAgg(plt.gcf())
        # 绘制图像
        canvas.draw()
        # 获取图像尺寸
        w, h = canvas.get_width_height()
        # 解码string 得到argb图像
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        # 重构成w h 4(argb)图像
        buf.shape = (w, h, 4)
        # 转换为 RGBA
        buf = np.roll(buf, 3, axis=2)
        # 得到 Image RGBA图像对象
        spectral_curve_image = Image.frombytes("RGBA", (w, h), buf.tobytes())
        # 转换为numpy array rgba四通道数组
        spectral_curve_image_array = np.asarray(spectral_curve_image)
        # 转换为rgb图像
        spectral_curve_image_array = spectral_curve_image_array[:, :, :3]
        self.spectral_curve_image = spectral_curve_image_array
        # cv2.imshow('img', rgb_image)
