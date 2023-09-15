conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/LVIS-Inapinting
git pull

python wj_lvis_inpaint.py \
--data_root /DDN_ROOT/wjpeng/dataset/LVIS \
--model_checkpoint /discobox/wjpeng/weights/stable-difusion-2-inpaint