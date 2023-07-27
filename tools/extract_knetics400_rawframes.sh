#!/usr/bin/env bash

SRC_PATH='/dev/shm/k400'
OUT_PATH='/discobox/wjpeng/dataset/k400/rawframes'

echo "Begin raw frames (RGB only) generation for train set"
python build_rawframes.py \
--src_dir $SRC_PATH/train/ \
--out_dir $OUT_PATH/train/ \
--level 2  \
--ext mp4 \
--task rgb  \
--new-short 256
echo "Raw frames (RGB only) generated for train set"

echo "Begin raw frames (RGB only) generation for val set"
python build_rawframes.py \
--src_dir $SRC_PATH/val \
--out_dir $OUT_PATH/val/ \
--level 2 \
--ext mp4 \
--task rgb  \
--new-short 256
echo "Raw frames (RGB only) generated for val set"
