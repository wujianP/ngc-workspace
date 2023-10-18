conda activate /discobox/wjpeng/env/xpaste
cd /discobox/wjpeng/code/202306/ngc-workspace/XPaste
git pull

cd generation
python text2im.py --model diffusers --samples 100 \
--category_file /DDN_ROOT/wjpeng/dataset/LVIS/annotations/coco/instances_train2017.json \
--output_dir /DDN_ROOT/wjpeng/dataset/instance_pool/LVIS_gen_FG