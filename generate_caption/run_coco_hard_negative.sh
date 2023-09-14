conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/generate_caption
git pull

python coco_hard_negative.py \
--model-path lmsys/fastchat-t5-3b-v1.0 \
--images_path /DDN_ROOT/wjpeng/dataset/coco2014/train2014 \
--annotations_path /DDN_ROOT/wjpeng/dataset/coco2014/annotations/captions_train2014.json \
--batch-size 8 \
--gpus 0 \
--outputs /DDN_ROOT/wjpeng/dataset/coco2014/annotations/
--job_num
--job_id
--save_freq