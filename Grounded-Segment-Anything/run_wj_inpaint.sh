# >>>>>>>> Visual Genome >>>>>>>>>>
conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything

export CUDA_VISIBLE_DEVICES=0
python wj_inpaint_visual_genome.py \
  --sam_hq_checkpoint /discobox/wjpeng/weights/hq-sam/sam_hq_vit_l.pth \
  --use_sam_hq \
  --output_dir "outputs" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --batch_size 4 \
  --num_workers 8


# >>>>>>>> RAM on Visual Genome >>>>>>>>>>
conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything

export CUDA_VISIBLE_DEVICES=0
python wj_inpaint_ram.py \
  --data_root /DDN_ROOT/wjpeng/dataset/coco2014/val2014 \
  --data_ann /DDN_ROOT/wjpeng/dataset/coco2014/annotations/captions_val2014.json \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
  --grounded_checkpoint /discobox/wjpeng/weights/groundingDINO/groundingdino_swinb_cogcoor.pth \
  --ram_checkpoint /discobox/wjpeng/weights/ram/ram_swin_large_14m.pth \
  --sam_hq_checkpoint /discobox/wjpeng/weights/hq-sam/sam_hq_vit_l.pth \
  --use_sam_hq \
  --output_dir "outputs" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --batch_size 8 \
  --num_workers 8


# >>>>>>>> Tag2Text on Visual Genome >>>>>>>>>>
conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything

export CUDA_VISIBLE_DEVICES=0
python wj_inpaint_tag2text.py \
  --lama_checkpoint /discobox/wjpeng/weights/big-lama \
  --lama_config /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything/lama/configs/prediction/default.yaml \
  --data_root /DDN_ROOT/wjpeng/dataset/visual-genome \
  --delete_tag_index /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything/wjpeng/preprocess_tags/tag2text/excluded_tag_indices.npy \
  --clustered_tags /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything/wjpeng/preprocess_tags/tag2text/clustered_tags.npy \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
  --grounded_checkpoint /discobox/wjpeng/weights/groundingDINO/groundingdino_swinb_cogcoor.pth \
  --tag2text_checkpoint /discobox/wjpeng/weights/tag2text/tag2text_swin_14m.pth \
  --sam_hq_checkpoint /discobox/wjpeng/weights/hq-sam/sam_hq_vit_l.pth \
  --sd_inpaint_checkpoint /discobox/wjpeng/weights/stable-difusion-2-inpaint \
  --use_sam_hq \
  --output_dir /DDN_ROOT/wjpeng/dataset/inpainted-visual-genome/stable-diffusion \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --iou_threshold 0.5 \
  --tag2text_threshold 0.64 \
  --batch_size 4 \
  --num_workers 8 \
  --visualize_freq 20
