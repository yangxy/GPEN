import os
import torch
import numpy as np
from rrdbnet_arch import RRDBNet
from torch.nn import functional as F

class RealESRNet(object):
    def __init__(self, base_dir='./', model=None, scale=2, device='cuda'):
        self.base_dir = base_dir
        self.scale = scale
        self.device = device
        self.load_srmodel(base_dir, model)

    def load_srmodel(self, base_dir, model):
        self.srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=23, num_grow_ch=32, scale=self.scale)
        if model is None:
            loadnet = torch.load(os.path.join(self.base_dir, 'weights', 'rrdb_realesrnet_psnr.pth'))
        else:
            loadnet = torch.load(os.path.join(self.base_dir, 'weights', model+'.pth'))
        #print(loadnet['params_ema'].keys)
        self.srmodel.load_state_dict(loadnet['params_ema'], strict=True)
        self.srmodel.eval()
        self.srmodel = self.srmodel.to(self.device)

    def process(self, img):
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        if self.scale == 2:
            mod_scale = 2
        elif self.scale == 1:
            mod_scale = 4
        else:
            mod_scale = None
        if mod_scale is not None:
            h_pad, w_pad = 0, 0
            _, _, h, w = img.size()
            if (h % mod_scale != 0):
                h_pad = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                w_pad = (mod_scale - w % mod_scale)
            img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')

        try:
            with torch.no_grad():
                output = self.srmodel(img)
            # remove extra pad
            if mod_scale is not None:
                _, _, h, w = output.size()
                output = output[:, :, 0:h - h_pad, 0:w - w_pad]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            return output
        except:
            return None