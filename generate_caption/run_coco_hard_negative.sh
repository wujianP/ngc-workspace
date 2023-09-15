# >> on total 400k captions
conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/generate_caption
git pull

JOB_ID=0

python coco_hard_negative.py \
--job_id $JOB_ID \
--gpus $((JOB_ID % 8)) \
--job_num 8 \
--model-path 	lmsys/fastchat-t5-3b-v1.0 \
--images_path /DDN_ROOT/wjpeng/dataset/coco2014/train2014 \
--annotations_path /DDN_ROOT/wjpeng/dataset/coco2014/annotations/captions_train2014.json \
--batch-size 32 \
--outputs /DDN_ROOT/wjpeng/dataset/coco2014/annotations/negative_caption_fastchat \
--save_freq 50


# >> on negCLIP 100k captions
conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/generate_caption
git pull

JOB_ID=0
export CUDA_VISIBLE_DEVICES=$((JOB_ID % 8))
python negCLIP_coco_hard_negative.py \
--job_id $JOB_ID \
--job_num 8 \
--model-path 	lmsys/fastchat-t5-3b-v1.0 \
--images_path /DDN_ROOT/wjpeng/dataset/coco2014/train2014 \
--annotations_path /discobox/wjpeng/code/202306/fineCLIP/Bow/temp_data/train_neg_clip.tsv \
--batch-size 32 \
--outputs /DDN_ROOT/wjpeng/dataset/coco2014/annotations/100k/negative_caption_fastchat \
--save_freq 50

# >> on negCLIP 100k captions Not FastChat
conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/generate_caption
git pull

JOB_ID=0
export CUDA_VISIBLE_DEVICES=$((JOB_ID % 8))
python negCLIP_coco_hard_negative.py \
--job_id $JOB_ID \
--job_num 8 \
--model-path 	THUDM/chatglm2-6b \
--images_path /DDN_ROOT/wjpeng/dataset/coco2014/train2014 \
--annotations_path /discobox/wjpeng/code/202306/fineCLIP/Bow/temp_data/train_neg_clip.tsv \
--batch-size 8 \
--outputs /DDN_ROOT/wjpeng/dataset/coco2014/annotations/100k/negative_caption_chatglm2-6b \
--save_freq 50

