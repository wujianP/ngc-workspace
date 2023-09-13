conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything

export CUDA_VISIBLE_DEVICES=0
python wj_image_caption.py \
--data_root /DDN_ROOT/wjpeng/dataset/inpainted-visual-genome/remove-stable-diffusion \
--data_ann wjpeng/ann/remove-stable-diffusion-output-eval.json \
--model_checkpoint /discobox/wjpeng/weights/blip2 \
--outputs /DDN_ROOT/wjpeng/dataset/inpainted-visual-genome/remove-stable-diffusion-filtered