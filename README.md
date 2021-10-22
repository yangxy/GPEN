# GAN Prior Embedded Network for Blind Face Restoration in the Wild

[Paper](https://arxiv.org/abs/2105.06070) | [Supplementary](https://www4.comp.polyu.edu.hk/~cslzhang/paper/GPEN-cvpr21-supp.pdf) | [Demo](https://vision.aliyun.com/experience/detail?spm=a211p3.14020179.J_7524944390.17.66cd4850wVDkUQ&tagName=facebody&children=EnhanceFace)

[Tao Yang](https://cg.cs.tsinghua.edu.cn/people/~tyang)<sup>1</sup>, Peiran Ren<sup>1</sup>, Xuansong Xie<sup>1</sup>, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>1,2</sup>  
_<sup>1</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com), Hangzhou, China_  
_<sup>2</sup>[Department of Computing, The Hong Kong Polytechnic University](http://www.comp.polyu.edu.hk), Hong Kong, China_

#### Face Restoration

<img src="figs/real_00.png" width="390px"/> <img src="figs/real_01.png" width="390px"/>
<img src="figs/real_02.png" width="390px"/> <img src="figs/real_03.png" width="390px"/>

<img src="figs/Solvay_conference_1927_comp.jpg" width="784px"/>

#### Face Colorization

<img src="figs/colorization_00.jpg" width="390px"/> <img src="figs/colorization_01.jpg" width="390px"/>

#### Face Inpainting

<img src="figs/inpainting_00.jpg" width="390px"/> <img src="figs/inpainting_01.jpg" width="390px"/>

#### Conditional Image Synthesis (Seg2Face)

<img src="figs/seg2face_00.jpg" width="390px"/> <img src="figs/seg2face_01.jpg" width="390px"/>

## News
(2021-07-06) The training code will be released soon. Stay tuned.

(2021-10-11) The Colab demo for GPEN is available now <a href="https://colab.research.google.com/drive/1fPUsJCpQipp2Z5B5GbEXqpBGsMp-nvjm?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>.

(2021-10-22) GPEN can now work with SR methods. A SR model trained by myself is provided. Replace it with your own model if necessary.

## Usage

![python](https://img.shields.io/badge/python-v3.7.4-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/pytorch-v1.7.0-green.svg?style=plastic)
![cuda](https://img.shields.io/badge/cuda-v10.2.89-green.svg?style=plastic)
![driver](https://img.shields.io/badge/driver-v460.73.01-green.svg?style=plastic)
![gcc](https://img.shields.io/badge/gcc-v7.5.0-green.svg?style=plastic)

- Clone this repository:
```bash
git clone https://github.com/yangxy/GPEN.git
cd GPEN
```
- Download RetinaFace model and our pre-trained model (not our best model due to commercial issues) and put them into ``weights/``.

    [RetinaFace-R50](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth) | [GPEN-BFR-512](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth) | [GPEN-BFR-512-D](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512-D.pth) | [GPEN-BFR-256](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-256.pth) | [GPEN-Colorization-1024](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Colorization-1024.pth) | [GPEN-Inpainting-1024](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Inpainting-1024.pth) | [GPEN-Seg2face-512](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Seg2face-512.pth) | [rrdb_realesrnet_psnr](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/rrdb_realesrnet_psnr.pth)

- Restore face images:
```bash
python face_enhancement.py --model GPEN-BFR-512 --size 512 --channel_multiplier 2 --narrow 1 --use_sr --indir examples/imgs --outdir examples/outs-BFR
```

- Colorize faces:
```bash
python face_colorization.py
```

- Complete faces:
```bash
python face_inpainting.py
```

- Synthesize faces:
```bash
python segmentation2face.py
```

## Main idea
<img src="figs/architecture.png" width="784px"/> 

## Citation
If our work is useful for your research, please consider citing:

    @inproceedings{Yang2021GPEN,
	    title={GAN Prior Embedded Network for Blind Face Restoration in the Wild},
	    author={Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang},
	    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	    year={2021}
    }
    
## License
© Alibaba, 2021. For academic and non-commercial use only.

## Acknowledgments
We borrow some codes from [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

## Contact
If you have any questions or suggestions about this paper, feel free to reach me at yangtao9009@gmail.com.
