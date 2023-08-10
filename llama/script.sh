conda activate /discobox/wjpeng/env/openVCLIP
cd /discobox/wjpeng/code/202306/ngc-workspace/llama
torchrun --nproc_per_node 1 --master_port 29500 generate_coco_hard_negative.py \
--gpu 0 \
--batch_size 32 \
--max_batch_size 32 \
--output_dir /discobox/wjpeng/dataset/coco2014/hardNegatives/train \
--save_freq 500 \
--prompt "In this task, you are required to modify the given sentence and return five variations of the sentence with different modifications. You should ensure that the objects (nouns) in the sentence remain as unchanged as possible, only altering the attributes of the objects (adjectives, adverbs) or the relationships between the objects (prepositions, word order). It's important to maintain a level of consistency between the modified sentences and the original one, avoiding significant differences." \
--filename coco2014train_hardNegative_llama-2-7b-chat.pth
