# GAN Prior Embedded Network for Blind Face Restoration in the Wild

[Paper](https://arxiv.org/abs/2105.06070) | [Supplementary](https://www4.comp.polyu.edu.hk/~cslzhang/paper/GPEN-cvpr21-supp.pdf) | [Demo](https://vision.aliyun.com/experience/detail?spm=a211p3.14020179.J_7524944390.17.66cd4850wVDkUQ&tagName=facebody&children=EnhanceFace)

<a href="https://replicate.ai/yangxy/gpen"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=blue"></a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/GPEN) 

[Tao Yang](https://cg.cs.tsinghua.edu.cn/people/~tyang)<sup>1</sup>, Peiran Ren<sup>1</sup>, Xuansong Xie<sup>1</sup>, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>1,2</sup>  
_<sup>1</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com), Hangzhou, China_  
_<sup>2</sup>[Department of Computing, The Hong Kong Polytechnic University](http://www.comp.polyu.edu.hk), Hong Kong, China_

#### Face Restoration

<img src="figs/real_00.png" width="390px"/> <img src="figs/real_01.png" width="390px"/>
<img src="figs/real_02.png" width="390px"/> <img src="figs/real_03.png" width="390px"/>

<img src="figs/Solvay_conference_1927_comp.jpg" width="784px"/>

#### Selfie Restoration

<img src="figs/selfie_00.jpg" width="390px"/> <img src="figs/selfie_01.jpg" width="390px"/>

#### Face Colorization

<img src="figs/colorization_00.jpg" width="390px"/> <img src="figs/colorization_01.jpg" width="390px"/>

#### Face Inpainting

<img src="figs/inpainting_00.jpg" width="390px"/> <img src="figs/inpainting_01.jpg" width="390px"/>

#### Conditional Image Synthesis (Seg2Face)

<img src="figs/seg2face_00.jpg" width="390px"/> <img src="figs/seg2face_01.jpg" width="390px"/>

## News
(2022-05-16) Add x1 sr model. Add ``--tile_size`` to avoid OOM.

(2022-03-15) Add x4 sr model. Try ``--sr_scale``.
 
(2022-03-09) Add GPEN-BFR-2048 for selfies. <font color='red'>I have to take it down due to commercial issues. Sorry about that.</font>

(2021-12-29) Add online demos <a href="https://replicate.ai/yangxy/gpen"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=blue"></a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/GPEN). Many thanks to [CJWBW](https://github.com/CJWBW) and [AK391](https://github.com/AK391).

(2021-12-16) Release a simplified training code of GPEN. It differs from our implementation in the paper, but could achieve comparable performance. We strongly recommend to change the degradation model.

(2021-12-09) Add face parsing to better paste restored faces back. 

(2021-12-09) GPEN can run on CPU now by simply discarding ``--use_cuda``. 

(2021-12-01) GPEN can now work on a Windows machine without compiling cuda codes. Please check it out. Thanks to [Animadversio](https://github.com/rosinality/stylegan2-pytorch/issues/81). Alternatively, you can try [GPEN-Windows](https://drive.google.com/file/d/1YJJVnPGq90e_mWZxSGGTptNQilZNfOEO/view?usp=drivesdk). Many thanks to [Cioscos](https://github.com/yangxy/GPEN/issues/74).

(2021-10-22) GPEN can now work with SR methods. A SR model trained by myself is provided. Replace it with your own model if necessary.

(2021-10-11) The Colab demo for GPEN is available now <a href="https://colab.research.google.com/drive/1fPUsJCpQipp2Z5B5GbEXqpBGsMp-nvjm?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>.

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

    [RetinaFace-R50](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116085&Signature=GlUNW6%2B8FxvxWmE9jKIZYOOciKQ%3D) | [ParseNet-latest](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116134&Signature=bnMwU1JogmNbARto6G%2B7iaJQCHs%3D) | [model_ir_se50](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/model_ir_se50.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116170&Signature=jEyBslytwpWoh5DfKvYe2H31GgE%3D) | [GPEN-BFR-512](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116208&Signature=hBgvVvKVSNGeXqT8glG%2Bd2t2OKc%3D) | [GPEN-BFR-512-D](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512-D.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116234&Signature=mP7MvYhKjbsIM2lhmuaEysssWpc%3D) | [GPEN-BFR-256](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-256.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116259&Signature=kMGJLSHqnvzzzqwtjUVBgngzX2s%3D) | [GPEN-BFR-256-D](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-256-D.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116288&Signature=b7NCfHFzyqKh%2BfaLrRCwMIIZ2HA%3D) | [GPEN-Colorization-1024](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Colorization-1024.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116315&Signature=9tPavW2h%2F1LhIKiXj73sTQoWqcc%3D) | [GPEN-Inpainting-1024](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Inpainting-1024.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116338&Signature=tvYhdLaLgW7UdcUrApXp2jsek8w%3D) | [GPEN-Seg2face-512](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Seg2face-512.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116362&Signature=VOaHmjFy5YVBjMoNTpVk2KDJx9k%3D) | [realesrnet_x1](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/realesrnet_x1.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1968049923&Signature=omV%2Fb8Jibkgl1FggsR%2B821jQvOI%3D) | [realesrnet_x2](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/realesrnet_x2.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1962694780&Signature=lI%2FolhA%2FyigiTRvoDIVbtMIyhjI%3D) | [realesrnet_x4](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/realesrnet_x4.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1962694847&Signature=MA5E%2FLp88oCz4kFINWdmeuSh7c4%3D)

- Restore face images:
```bash
python demo.py --task FaceEnhancement --model GPEN-BFR-512 --in_size 512 --channel_multiplier 2 --narrow 1 --use_sr --sr_scale 4 --use_cuda --save_face --indir examples/imgs --outdir examples/outs-bfr
```

- Colorize faces:
```bash
python demo.py --task FaceColorization --model GPEN-Colorization-1024 --in_size 1024 --use_cuda --indir examples/grays --outdir examples/outs-colorization
```

- Complete faces:
```bash
python demo.py --task FaceInpainting --model GPEN-Inpainting-1024 --in_size 1024 --use_cuda --indir examples/ffhq-10 --outdir examples/outs-inpainting
```

- Synthesize faces:
```bash
python demo.py --task Segmentation2Face --model GPEN-Seg2face-512 --in_size 512 --use_cuda --indir examples/segs --outdir examples/outs-seg2face
```

- Train GPEN for BFR with 4 GPUs:
```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train_simple.py --size 1024 --channel_multiplier 2 --narrow 1 --ckpt weights --sample results --batch 2 --path your_path_of_croped+aligned_hq_faces (e.g., FFHQ)

```
When testing your own model, set ``--key g_ema``.

Please check out ``run.sh`` for more details.

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
We borrow some codes from [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), and [GFPGAN](https://github.com/TencentARC/GFPGAN).

## Contact
If you have any questions or suggestions about this paper, feel free to reach me at yangtao9009@gmail.com.
