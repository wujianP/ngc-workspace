cd /discobox/wjpeng/code/202306/ngc-workspace/llama
torchrun --nproc_per_node 1 generate_coco_hard_negative.py \
--batch_size 32 \
--max_batch_size 32 \
--output_dir /discobox/wjpeng/dataset/coco2014/hardNegatives/train \
--save_freq 500 \
--prompt "We will input a sentence, and you need to modify one word or phrase in it (which can be an adjective, verb, preposition, adverb, or quantifier) to change the meaning of the sentence that is described. If there are multiple words of the same part of speech in the sentence, you can swap their positions. It's important to make minimal changes to keep the overall sentence unchanged, and try not to modify nouns if possible. Five differently modified sentences are required for each sentence." \
--filename coco2014train_hardNegative_llama-2-7b-chat.pth
