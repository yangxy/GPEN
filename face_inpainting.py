'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import math
import numpy as np
from PIL import Image, ImageDraw
import __init_paths
from face_model.face_gan import FaceGAN

# modified by yangxy
def brush_stroke_mask(img, color=(255,255,255)):
    min_num_vertex = 8
    max_num_vertex = 28
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('RGB', (W, H), 0)
        if img is not None: mask = img #Image.fromarray(img)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=color, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

class FaceInpainting(object):
    def __init__(self, base_dir='./', size=1024, model=None, channel_multiplier=2):
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, brokenf):
        # complete the face
        out = self.facegan.process(brokenf)

        return out

if __name__=='__main__':
    model = {'name':'GPEN-Inpainting-1024', 'size':1024}
    
    indir = 'examples/ffhq-10'
    outdir = 'examples/outs-inpainting'
    os.makedirs(outdir, exist_ok=True)

    faceinpainter = FaceInpainting(size=model['size'], model=model['name'], channel_multiplier=2)

    files = sorted(glob.glob(os.path.join(indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)
        
        originf = cv2.imread(file, cv2.IMREAD_COLOR)
        
        brokenf = np.asarray(brush_stroke_mask(Image.fromarray(originf)))

        completef = faceinpainter.process(brokenf)
        
        originf = cv2.resize(originf, completef.shape[:2])
        brokenf = cv2.resize(brokenf, completef.shape[:2])
        cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1])+'.jpg'), np.hstack((brokenf, completef, originf)))
        
        if n%10==0: print(n, file)
        
