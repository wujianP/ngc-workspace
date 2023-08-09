torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir /discobox/wjpeng/weights/llama2/llama-2-7b-chat/ \
    --tokenizer_path /discobox/wjpeng/weights/llama2/tokenizer.model \
    --max_seq_len 512 --max_batch_size 16