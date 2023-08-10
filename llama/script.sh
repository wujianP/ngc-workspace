torchrun --nproc_per_node 1 generate_coco_hard_negative.py \
--batch_size 16 \
--output_dir /discobox/wjpeng/dataset/coco2014/hardNegatives/train \
--save_freq 5 \
--prompt "We will input a sentence, and you need to modify one word or phrase in it (it can be a noun, adjective, verb, or quantifier) to change the meaning of the sentence that is described. If there are multiple words of the same part of speech in the sentence, you can swap their positions. It's important to make minimal changes to keep the overall sentence unchanged. Five differently modified sentences are required for each sentence." \
--filename coco2014train_hardNegative_llama-2-7b-chat.pth
