WEIGHT_PATH=/discobox/wjpeng/weights
cd /discobox/wjpeng/code/202306/ngc-workspace/segment_video

export CUDA_VISIBLE_DEVICES=0
python groundedeSAM.py \
  --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint $WEIGHT_PATH/groundingDINO/groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint $WEIGHT_PATH/hq-sam/sam_hq_vit_b.pth \
  --use_sam_hq \
  --input_image /discobox/wjpeng/code/202306/ngc-workspace/segment_video/demo_images/12.png \
  --output_dir /discobox/wjpeng/code/202306/ngc-workspace/segment_video/outputs \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt 'people.music instrument.animal.ball.tool.'