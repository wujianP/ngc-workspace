OUTPUT_DIR='.'
PROMPT='We will input a sentence, and what you need to do is make minor changes to the sentence to alter its meaning. Each time, please output 4 modified sentences in the form of a list. Avoid providing unrelated content.'

torchrun --nproc_per_node 1 generate_coco_hard_negative.py \
--output_dir $OUTPUT_DIR \
--prompt $PROMPT

torchrun --nproc_per_node 1 generate_coco_hard_negative.py \
--output_dir . \
--prompt test