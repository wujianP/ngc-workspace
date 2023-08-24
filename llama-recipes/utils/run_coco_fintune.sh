cd /discobox/wjpeng/code/202306/ngc-workspace/llama-recipes
python llama_finetuning.py  \
--use_peft \
--peft_method lora \
--model_name /discobox/wjpeng/weights/llama2-7b-hf \
--output_dir /discobox/wjpeng/weights/llama2-7b-coco \
--num_epochs 1 \
--dataset coco_caption_dataset \
--batch_size_training 1 \
--lr 1e-5
