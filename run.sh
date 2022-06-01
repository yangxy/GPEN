############## TEST ################
#python demo.py --task FaceEnhancement --model GPEN-BFR-512 --in_size 512 --channel_multiplier 2 --narrow 1 --use_sr --sr_scale 2 --use_cuda --save_face --indir examples/imgs --outdir examples/outs-bfr
#python demo.py --task FaceEnhancement --model GPEN-BFR-256 --in_size 256 --channel_multiplier 1 --narrow 0.5 --use_sr --sr_scale 4 --use_cuda --save_face --indir examples/imgs --outdir examples/outs-bfr
python demo.py --task FaceEnhancement --model GPEN-BFR-2048 --in_size 2048 --channel_multiplier 2 --narrow 1 --alpha 0.8 --use_sr --sr_scale 2 --use_cuda --tile_size 400 --save_face --indir examples/selfie --outdir examples/outs-selfie

#python demo.py --task FaceColorization --model GPEN-Colorization-1024 --in_size 1024 --use_cuda --indir examples/tmp --outdir examples/outs-colorization

#python demo.py --task FaceInpainting --model GPEN-Inpainting-1024 --in_size 1024 --use_cuda --indir examples/ffhq-10 --outdir examples/outs-inpainting

#python demo.py --task Segmentation2Face --model GPEN-Seg2face-512 --in_size 512 --use_cuda --indir examples/segs --outdir examples/outs-seg2face

############## TRAIN ################
#CUDA_VISIBLE_DEVICES='0,1,' python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train_simple.py --size 512 --channel_multiplier 2 --narrow 1 --ckpt ckpt-512 --sample sample-512 --batch 1 --path your_path_of_aligned_faces
