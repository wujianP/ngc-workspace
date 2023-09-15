conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/LVIS-Inapinting
git pull

export CUDA_VISIBLE_DEVICES 1
python wj_lvis_inpaint.py \
--data_root /DDN_ROOT/wjpeng/dataset/LVIS \
--ann /DDN_ROOT/wjpeng/dataset/LVIS/lvis_v1_val.json \
--model_checkpoint /discobox/wjpeng/weights/stable-difusion-2-inpaint
