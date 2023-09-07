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
  --batch_size 4 \
  --num_workers 8


# >>>>>>>> Tag2Text on Visual Genome >>>>>>>>>>
conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything

export CUDA_VISIBLE_DEVICES=0
python wj_inpaint_tag2text.py \
  --data_root /DDN_ROOT/wjpeng/dataset/coco2014/val2014 \
  --data_ann /DDN_ROOT/wjpeng/dataset/coco2014/annotations/captions_val2014.json \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
  --grounded_checkpoint /discobox/wjpeng/weights/groundingDINO/groundingdino_swinb_cogcoor.pth \
  --tag2text_checkpoint /discobox/wjpeng/weights/tag2text/tag2text_swin_14m.pth \
  --sam_hq_checkpoint /discobox/wjpeng/weights/hq-sam/sam_hq_vit_l.pth \
  --use_sam_hq \
  --output_dir "outputs" \
  --box_threshold 0.35 \
  --text_threshold 0.3 \
  --iou_threshold 0.5 \
  --batch_size 4 \
  --num_workers 8
