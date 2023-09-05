conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything

export CUDA_VISIBLE_DEVICES=0
python wj_inpaint_tag2text.py \
  --data_root /DDN_ROOT/wjpeng/dataset/coco2014/val2014 \
  --data_ann /DDN_ROOT/wjpeng/dataset/coco2014/annotations/captions_val2014.json \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /discobox/wjpeng/weights/groundingDINO/groundingdino_swint_ogc.pth \
  --tag2text_checkpoint /discobox/wjpeng/weights/tag2text/tag2text_swin_14m.pth \
  --sam_hq_checkpoint /discobox/wjpeng/weights/hq-sam/sam_hq_vit_h.pth \
  --use_sam_hq \
  --output_dir "outputs" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --batch_size 32 \
  --num_workers 8
