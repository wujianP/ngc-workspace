conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/llama
torchrun --nproc_per_node 1 --master_port 29500 wj_generate_hard_negative.py \
--gpu 0 \
--batch_size 32 \
--max_batch_size 32 \
--output_dir /discobox/wjpeng/dataset/coco2014/hardNegatives/train \
--save_freq 500 \
--prompt "In this task, you need to modify the given sentence to change its meaning. You can alter a word or phrase (noun, adjective, verb, preposition, etc.), or swap the positions of two words. Ensure that the sentence's meaning is altered, and guarantee that the modified sentences are not significantly different from the original one." \
--filename coco2014train_hardNegative_llama-2-7b-chat
