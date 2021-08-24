from __future__ import print_function
import sys
sys.path.append('../')
sys.path.append('../jai_package')
sys.path.append('../retinaface_torch')
sys.path.append('../retinaface_torch/weights')
import numpy as np
import cv2
import os
import torch
import torch.backends.cudnn as cudnn

from other_utils import face_align

from retinaface_torch.data import cfg_mnet, cfg_re50
from retinaface_torch.layers.functions.prior_box import PriorBox
from retinaface_torch.utils.nms.py_cpu_nms import py_cpu_nms
from retinaface_torch.utils.box_utils import decode, decode_landm
from retinaface_torch.models.retinaface import RetinaFace

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from system_gui import *
from jai_package import jai_camera_package


# retinaface
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
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
    '''
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        bbox = b[0:4]
        pts5 = b[]
    # save image

    name = "test.jpg"
    cv2.imwrite(name, img_raw)
    '''


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        # 用于描述rgb相机和单色相机对于人脸采集后识别的结果
        self.face_name_c = 'temp_rgb'
        self.face_score_c = 0
        self.face_name_m = 'temp_mono'
        self.face_score_m = 0
        self.face_texture = 'paper'
        self.face_texture_score = '0'

        # 用于获取 jai相机图像 以及 描述获取状态的变量
        self.jai_go5000C_camera = 0
        self.jai_go5000C_image = np.zeros((2048, 2560, 1), np.uint8)
        self.jai_go5000C_open_state = 1

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

    def open_camera_1(self):
        if not self.timer_camera_c.isActive():       # 若定时器未启动
            self.jai_go5000C_camera = jai_camera_package.JaiCamera(0)
            self.jai_go5000C_open_state = self.jai_go5000C_camera.open_camera()
            self.jai_go5000C_image = self.jai_go5000C_camera.get_raw_image()
            if not self.jai_go5000C_open_state:    # camera_open_state表示open()成不成功
                msg = QMessageBox.warning(self, 'warning', "请检查相机是否连接正确", buttons=QMessageBox.Ok)
            else:
                self.timer_camera_c.start(30)        # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.pushButton_open_camera_1.setText('关闭相机1')
        else:
            self.timer_camera_c.stop()  # 关闭定时器
            if self.jai_go5000C_open_state:
                self.jai_go5000C_camera.close_camera()
            self.jai_go5000C_open_state = 0
            self.label_show_rgb_image.clear()  # 清空视频显示区域
            self.pushButton_open_camera_1.setText('打开相机1')
            self.init_rgb_face_recognition()

    def show_camera_c(self):
        go5000_image = cv2.cvtColor(self.jai_go5000C_image, cv2.COLOR_BAYER_GR2BGR)
        go5000_image = cv2.resize(go5000_image, (0, 0), fx=0.1, fy=0.1)
        if self.jai_go5000C_open_state:
            try:
                bbox_pts = retinaface_detect(self.retinaface_net, go5000_image)[0]
                #print(bbox_pts)
                face_current_bbox = bbox_pts[0:4]
                face_current_pts5 = np.zeros((5, 2))
                for i in range(5):
                    for j in range(2):
                        face_current_pts5[i, j] = bbox_pts[5+2*i+j] 
                #print(face_current_pts5.shape)
                align_face_img = face_align.norm_crop(go5000_image, face_current_pts5)
                showImage = QtGui.QImage(align_face_img.data, align_face_img.shape[1], align_face_img.shape[0], align_face_img.shape[1] * 3, QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_show_rgb_align_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
                cv2.rectangle(go5000_image, (int(face_current_bbox[0]), int(face_current_bbox[1])),(int(face_current_bbox[2]), int(face_current_bbox[3])), (0, 255, 0), 1)
            except:
                pass
            showImage = QtGui.QImage(go5000_image.data, go5000_image.shape[1],
                                     go5000_image.shape[0], go5000_image.shape[1] * 3,
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_rgb_image.setPixmap(QtGui.QPixmap.fromImage(showImage))


    def select_face_1(self):
        pass

    def collect_face_1(self):
        pass

    def start_recognition_1(self):
        pass

    def select_font(self):
        font, ok = QtWidgets.QFontDialog.getFont()
        if ok:
            self.label_text_name_left.setFont(font)
            self.label_text_conf_left.setFont(font)
            self.label_text_conf_right.setFont(font)
            self.label_text_name_right.setFont(font)

    def init_system(self):
        pass

    def system_exit(self):
        pass

    def open_camera_2(self):
        pass

    def show_camera_m(self):
        pass

    def select_face_2(self):
        pass

    def collect_face_2(self):
        pass

    def start_recognition_2(self):
        pass

    def init_rgb_face_recognition(self):
        pass

    def init_mono_face_recognition(self):
        pass

