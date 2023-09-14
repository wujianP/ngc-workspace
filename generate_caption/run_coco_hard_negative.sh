conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/generate_caption
git pull

JOB_ID=0

python coco_hard_negative.py \
--job_id $JOB_ID \
--gpus $((JOB_ID % 8)) \
--job_num 16 \
--model-path lmsys/fastchat-t5-3b-v1.0 \
--images_path /DDN_ROOT/wjpeng/dataset/coco2014/train2014 \
--annotations_path /DDN_ROOT/wjpeng/dataset/coco2014/annotations/captions_train2014.json \
--batch-size 16 \
--outputs /DDN_ROOT/wjpeng/dataset/coco2014/annotations/ \
--save_freq 50
