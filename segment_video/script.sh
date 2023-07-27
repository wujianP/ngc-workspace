python k400_Inference.py \
--model_path /discobox/wjpeng/weights/fastSAM/FastSAM-x.pt \
--data_path /dev/shm/k400/ \
--ann_path /discobox/wjpeng/dataset/k400/ann/train.csv


python Inference.py \
--model_path /discobox/wjpeng/weights/fastSAM/FastSAM-x.pt \
 --img_path /discobox/wjpeng/code/202306/ngc-workspace/segment_video/FastSAM/data/img1.png  \
 --text_prompt "the yellow dog"