'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import cv2
from face_model.face_gan import FaceGAN

class FaceColorization(object):
    def __init__(self, base_dir='./', in_size=1024, out_size=1024, model=None, channel_multiplier=2, narrow=1, key=None, device='cuda'):
        self.facegan = FaceGAN(base_dir, in_size, out_size, model, channel_multiplier, narrow, key, device=device)

    def post_process(self, gray, out):
        out_rs = cv2.resize(out, gray.shape[:2][::-1])
        gray_yuv = cv2.cvtColor(gray, cv2.COLOR_BGR2YUV)
        out_yuv = cv2.cvtColor(out_rs, cv2.COLOR_BGR2YUV)

        out_yuv[:, :, 0] = gray_yuv[:, :, 0]
        final = cv2.cvtColor(out_yuv, cv2.COLOR_YUV2BGR)

        return final

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, gray, aligned=True):
        # colorize the face
        out = self.facegan.process(gray)

        if gray.shape[:2] != out.shape[:2]:
            out = self.post_process(gray, out)

        return out, [gray], [out]
        
        
