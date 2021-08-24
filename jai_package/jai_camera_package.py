import ctypes
import numpy as np

class JaiCamera:
    image_container = 0
    frame_pointer = 0
    lib_load = 0

    def __init__(self, camera_id):
        self.camera_id = camera_id
        if self.camera_id == 0:
            self.lib_load = ctypes.cdll.LoadLibrary('../jai_package/go5000C.so')
        else:
            self.lib_load = ctypes.cdll.LoadLibrary('../jai_package/go5100M.so')


    def open_camera(self):
        try:
            self.lib_load.start()
            return True
        except:
            return False

    def get_raw_image(self):
        if self.camera_id == 0:
            self.image_container=np.zeros((2048,2560,1),np.uint8)
        else:
            self.image_container = np.zeros((2056, 2464, 1), np.uint8)
        self.frame_pointer=np.asarray(self.image_container,dtype=np.uint8)
        self.frame_pointer=self.frame_pointer.ctypes.data_as(ctypes.c_char_p)
        self.lib_load.acquireImages(self.frame_pointer)
        return self.image_container

    def close_camera(self):
        self.lib_load.stop()
