#if running on multi-gpu machine
conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/llama-recipes
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes 1 --nproc_per_node 8 llama_finetuning.py  \
--dataset alpaca_dataset  \
--num_epochs 2 \
--num_workers_dataloader 8 \
--batch_size_training 4 \
--lr 1e-5 \
--use_peft \
--peft_method lora \
--quantization \
--model_name /discobox/wjpeng/weights/llama-2-7b-hf \
--output_dir /discobox/wjpeng/weights/llama-2-7b-tuned/winoground_eidt


python generate_coco.py \
    --base_model '/discobox/wjpeng/weights/llama-2-7b-hf' \
    --lora_weights '/discobox/wjpeng/weights/llama-2-7b-coco-lora'

import huggingface_hub
huggingface_hub.snapshot_download(
            "meta-llama/Llama-2-7b-hf",
            local_dir="/discobox/wjpeng/weights/llama-2-7b-hf",
            token="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"
        )