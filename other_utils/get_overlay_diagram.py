import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def get_overlay_image_from_hyperspectral(image_31):
    overlay_out_array = np.ones((image_31.shape[0], image_31.shape[1], 3))*255
    overlay_out_image = Image.fromarray(np.uint8(overlay_out_array))

    colors_cmap = plt.get_cmap('jet', 31)
    color_list = plt.get_cmap(colors_cmap, 31)([i for i in range(31)])

    for i in range(31):
        image_3_temp = np.zeros((image_31.shape[0], image_31.shape[1], 3))
        image_3_temp[:,:,0] = image_3_temp[:,:,1] = image_3_temp[:, :, 2] = image_31[:, :, i]
        image_3_temp = cv2.resize(image_3_temp, (0, 0), fx=0.5, fy=0.5)
        image_3_temp[:, :, 0] = image_3_temp[:, :, 0] * color_list[i, 2]
        image_3_temp[:, :, 1] = image_3_temp[:, :, 1] * color_list[i, 1]
        image_3_temp[:, :, 2] = image_3_temp[:, :, 2] * color_list[i, 0]
        image_3_temp_PIL = Image.fromarray(np.uint8(image_3_temp))
        overlay_out_image.paste(image_3_temp_PIL, (int(image_31.shape[1]/2) - i*int(image_31.shape[1]/62), i*int(image_31.shape[0]/62)))

    return np.asarray(overlay_out_image)


if __name__ == '__main__':
    print('get overlay hyperspectral image')
    #image_31 = np.ones((200, 200, 31))*255
    image_31 = np.random.randint(1, 255, size=(200, 200*31))
    image_31 = np.reshape(image_31, (200, 200, 31))
    #image_31 = cv2.resize(image_31, (1000, 1000))
    image_out = get_overlay_image_from_hyperspectral(image_31)
    image_out = cv2.resize(image_out, (500, 500))
    print(image_out.shape)
    cv2.imshow('temp', image_out)
    cv2.waitKey(-1)


