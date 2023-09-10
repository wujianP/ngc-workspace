conda activate /discobox/wjpeng/env/clip/
cd /discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything/wjpeng/preprocess_tags/tag2text

python cluster_tags.py \
--src_file='translated_tag_list.txt' \
--dest_file='clustered_tags'