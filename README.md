# Hyperspectral-face-recognition-gui-demo
Hyperspectral-face-recognition-gui-demo

## 1、人脸识别部分 insightface
Retinaface + Arcface

人脸检测   +  人脸识别
### (1)
系统中使用的Retinaface的backbone为resnet50 检测精度可达90%  速度：0.5-0.6s  (640p)
系统中使用的Arcface的backbone为resnet50（fp16）  速度：0.08-0.09s   (112p)

### (2)
Retinaface的backbone为mobelnet0.25 速度： 0.07-0.08s      (640p)
Arcface 的backbone为resnet18 （fp16）速度：0.03-0.04 s    (112P)


## 2、高光谱图像重建
Unet （fp32）
速度：0.05 s    (512p)


## 3、系统运行总帧数
0.08+0.03+0.05 = 0.16s
两个演示同时进行时运行时间为： （0.08+0.03）*2 + 0.05 = 0.27s
所以系统运行的总帧数大概为：3-5 FPS


## 4、TIPs
（1）若输入图片resize为256p，帧数大概可以达到 7-8FPS，但是需要识别的时候人脸占据整张图像面积较大，
才可以有较高的准确率。
（2）如果增加并行，可能可以使速度更快，还未尝试。
>>>>>>> bb0189c (the whole project release_v1.0)
