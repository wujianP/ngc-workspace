WEIGHT_PATH=/discobox/wjpeng/weights
DATA_PATH=/discobox/wjpeng/dataset/k400/ann/rawframe_list.txt
OUT_PATH=/discobox/wjpeng/dataset/k400/ann/groundedSAM/DINO-SwinT-imgSize800_SAMHQ-ViTB-imgSize1024_stride8.npy
cd /discobox/wjpeng/code/202306/ngc-workspace/segment_video

python groundedeSAM.py \
  --sample_stride 8 \
  --batch_size 8 \
  --sam_img_size 1024 \
  --grounding_dino_img_size 800 \
  --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint $WEIGHT_PATH/groundingDINO/groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint $WEIGHT_PATH/hq-sam/sam_hq_vit_b.pth \
  --use_sam_hq \
  --data_path $DATA_PATH \
  --output $OUT_PATH \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --visualize_freq 10 \
  --save_freq 10 \
  --text_prompt 'people.music instrument.animal.ball.tool.'
