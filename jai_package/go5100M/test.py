import ctypes
import numpy as np
import sys
import cv2

lib=ctypes.cdll.LoadLibrary('./libImageOut.so')
img=np.zeros((2056,2464,1),np.uint8)
frame_data=np.asarray(img,dtype=np.uint8)
frame_data=frame_data.ctypes.data_as(ctypes.c_char_p)

if sys.argv[1]=='start':
    lib.start()
     
lib.acquireImages(frame_data)
img = img.reshape(2056,2464,1)

#cv2.imwrite('test.jpg',convert_img)
while 1:
    convert_img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)  
    cv2.imshow('img',convert_img)
    if cv2.waitKey(1)&0xff == ord('q'):       
        lib.stop()
        cv2.destroyAllWindows()
        break



