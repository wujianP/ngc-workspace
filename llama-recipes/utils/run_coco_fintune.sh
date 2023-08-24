
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/llama-recipes

torchrun --nnodes 1 --nproc_per_node 8 llama_finetuning.py  \
--use_peft \
--peft_method prefix \
--model_name /discobox/wjpeng/weights/llama2-7b-hf \
--output_dir /discobox/wjpeng/weights/llama2-7b-coco \
--num_epochs 1 \
--dataset coco_caption_dataset \
--batch_size_training 1 \
--lr 1e-5 \
--enable_fsdp \
--quantization \
--use_fp16 \
--pure_bf16
