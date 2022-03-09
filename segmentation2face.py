'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
from face_model.face_gan import FaceGAN

class Segmentation2Face(object):
    def __init__(self, base_dir='./', in_size=1024, out_size=None, model=None, channel_multiplier=2, narrow=1, key=None, is_norm=True, device='cuda'):
        self.facegan = FaceGAN(base_dir, in_size, out_size, model, channel_multiplier, narrow, key, is_norm, device=device)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, segf, aligned=True):
        # from segmentations to faces
        out = self.facegan.process(segf)

        return out, [segf], [out]
    
        
