'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import numpy as np
from PIL import Image
import __init_paths
from face_model.face_gan import FaceGAN

class FaceColorization(object):
    def __init__(self, base_dir='./', size=1024, model=None, channel_multiplier=2):
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, gray):
        # colorize the face
        out = self.facegan.process(gray)

        return out
        

if __name__=='__main__':
    model = {'name':'GPEN-1024-Color', 'size':1024}
    
    indir = 'examples/grays'
    outdir = 'examples/couts'
    os.makedirs(outdir, exist_ok=True)

    facecolorizer = FaceColorization(size=model['size'], model=model['name'], channel_multiplier=2)

    files = sorted(glob.glob(os.path.join(indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)
        
        grayf = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        grayf = cv2.cvtColor(grayf, cv2.COLOR_GRAY2BGR) # channel: 1->3

        colorf = facecolorizer.process(grayf)
        
        grayf = cv2.resize(grayf, colorf.shape[:2])
        cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1])+'.jpg'), np.hstack((grayf, colorf)))
        
        if n%10==0: print(n, file)
        
