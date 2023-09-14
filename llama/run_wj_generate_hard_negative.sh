conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/llama
torchrun --nproc_per_node 1 --master_port 29500 wj_generate_hard_negative.py \
--gpu 0 \
--batch_size 32 \
--max_batch_size 32 \
--output_dir /discobox/wjpeng/dataset/coco2014/hardNegatives/train \
--save_freq 500 \
--prompt "In this task, I will give you some objects, and you will generate a sentence containing these objects." \
--filename coco2014train_hardNegative_llama-2-7b-chat
