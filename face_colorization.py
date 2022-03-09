'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
from face_model.face_gan import FaceGAN

class FaceColorization(object):
    def __init__(self, base_dir='./', in_size=1024, out_size=1024, model=None, channel_multiplier=2, narrow=1, key=None, device='cuda'):
        self.facegan = FaceGAN(base_dir, in_size, out_size, model, channel_multiplier, narrow, key, device=device)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, gray, aligned=True):
        # colorize the face
        out = self.facegan.process(gray)

        return out, [gray], [out]
        
        
