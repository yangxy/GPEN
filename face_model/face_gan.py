'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import torch
import os
import cv2
import glob
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, utils
from model import FullGenerator

class FaceGAN(object):
    def __init__(self, base_dir='./', size=512, model=None, channel_multiplier=2):
        self.mfile = os.path.join(base_dir, 'weights', model+'.pth')
        self.n_mlp = 8
        self.resolution = size
        self.load_model(channel_multiplier)

    def load_model(self, channel_multiplier=2):
        self.model = FullGenerator(self.resolution, 512, self.n_mlp, channel_multiplier).cuda()
        pretrained_dict = torch.load(self.mfile)
        self.model.load_state_dict(pretrained_dict)
        self.model.eval()

    def process(self, img):
        img = cv2.resize(img, (self.resolution, self.resolution))
        img_t = self.img2tensor(img)

        with torch.no_grad():
            out, __ = self.model(img_t)

        out = self.tensor2img(out)

        return out

    def img2tensor(self, img):
        img_t = (torch.from_numpy(img).cuda()/255. - 0.5) / 0.5
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).flip(1) # BGR->RGB
        return img_t

    def tensor2img(self, image_tensor, pmax=255.0, imtype=np.uint8):
        image_tensor = image_tensor * 0.5 + 0.5
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        image_numpy = np.clip(image_tensor.float().cpu().numpy(), 0, 1) * pmax

        return image_numpy.astype(imtype)
