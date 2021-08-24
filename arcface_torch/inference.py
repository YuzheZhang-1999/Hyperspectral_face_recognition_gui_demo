import argparse
import cv2
import numpy as np
import torch
import time

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    device = torch.device("cuda:0")
    print(device)
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(weight))
    net = net.to(device)
    net.eval()

    for i in range(5):
        start = time.time()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        img = img.to(device)
        feat = net(img)
        end = time.time()
        #print(feat)
        print('time is ' + str(end-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='ms1mv3_arcface_r50_fp16/backbone.pth')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)

