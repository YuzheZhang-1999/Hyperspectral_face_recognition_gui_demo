# 引入 FigureCanvasAgg
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
# 引入 Image
import PIL.Image as Image
# 将plt转化为numpy数据
import cv2
import time

'''
while True:
    spectral_curve_wiener = np.random.randn(10, 10, 31)
    spectral_curve_wiener = spectral_curve_wiener[0, 0, :]
    #print(spectral_curve_wiener)
    axis_x = np.linspace(420, 720, num=31)
    plt.clf()
    plt.title('Spectral curve')
    plt.plot(axis_x, spectral_curve_wiener)  # 显示wiener预测的光谱曲线
    # plt.savefig('test')
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
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    rgb_image = cv2.resize(rgb_image, (500, 500))
    cv2.imshow('img', rgb_image)
    print(rgb_image.shape)
    cv2.waitKey(1)
    #print(time.time())
'''
def show_spectral_curve_on_gui(spectral_curve_list):
    axis_x = np.linspace(420, 720, num=31)
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
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    spectral_curve_image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    # 转换为numpy array rgba四通道数组
    spectral_curve_image_array = np.asarray(spectral_curve_image)
    # 转换为rgb图像
    spectral_curve_image_array = spectral_curve_image_array[:, :, :3]
    plt.clf()
    plt.close()
    return spectral_curve_image_array
    #cv2.imshow('img', rgb_image)

if __name__ == '__main__':
    while True:
        array = np.random.randn(256, 256, 31)
        print(array)
        image = show_spectral_curve_on_gui((array[int(array.shape[0] / 2), int(array.shape[1] / 2), :]).tolist())
        cv2.imshow('img', image)
        cv2.waitKey(1)