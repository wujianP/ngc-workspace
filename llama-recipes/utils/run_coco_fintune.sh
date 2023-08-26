CUDA_VISIBLE_DEVICES=0,1,2,3

conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/llama-recipes

torchrun --nnodes 1 --nproc_per_node 4 llama_finetuning.py  \
--use_peft \
--peft_method prefix \
--model_name /discobox/wjpeng/weights/llama-2-7b-hf \
--output_dir /discobox/wjpeng/weights/llama2-7b-coco \
--num_epochs 1 \
--dataset coco_caption_dataset \
--batch_size_training 4 \
--lr 1e-5 \
--quantization


#######################################
#if running on multi-gpu machine
conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/llama-recipes
export CUDA_VISIBLE_DEVICES=0

python llama_finetuning.py  \
--dataset alpaca_dataset  \
--num_epochs 1 \
--lr 1e-5 \
--use_peft \
--peft_method lora \
--quantization \
--model_name /discobox/wjpeng/weights/llama-2-7b-hf \
--output_dir /discobox/wjpeng/weights/llama-2-7b-coco-lora


python generate.py \
    --base_model '/discobox/wjpeng/weights/llama-2-7b-hf' \
    --lora_weights '/discobox/wjpeng/weights/llama-2-7b-coco-lora'
