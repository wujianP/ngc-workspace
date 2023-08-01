WEIGHT_PATH=/discobox/wjpeng/weights
DATA_PATH=/discobox/wjpeng/dataset/k400/ann/rawframe_list.txt
OUT_PATH=/discobox/wjpeng/code/202306/ngc-workspace/segment_video/outputs
cd /discobox/wjpeng/code/202306/ngc-workspace/segment_video

python groundedeSAM.py \
  --batch_size 8 \
  --num_worker 8 \
  --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint $WEIGHT_PATH/groundingDINO/groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint $WEIGHT_PATH/hq-sam/sam_hq_vit_b.pth \
  --use_sam_hq \
  --data_path $DATA_PATH \
  --output_dir $OUT_PATH \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt 'people.music instrument.animal.ball.tool.'
