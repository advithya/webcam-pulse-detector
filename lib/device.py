import cv2, time
import  base64  # Add base64 import
import numpy as np
import urllib.request  # Add this import

class ipCamera(object):

    def __init__(self,url, user = None, password = None):
        self.url = url
        auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]

        self.req = urllib.request.Request(self.url)  # Update urllib2 to urllib.request
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib.request.urlopen(self.req)  # Update urllib2 to urllib.request
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame

class Camera(object):

    def __init__(self, camera = 0):
        self.cam = cv2.VideoCapture(camera)
        self.valid = False
        try:
            resp = self.cam.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None

    def get_frame(self):
        if self.valid:
            _,frame = self.cam.read()
        else:
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def release(self):
        self.cam.release()