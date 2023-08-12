#!/bin/bash

# Make data folder relative to script location
#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

OUT_DIR="/discobox/wjpeng/dataset/pix2pix"

mkdir -p $OUT_DIR

# Copy text datasets
wget -q --show-progress http://instruct-pix2pix.eecs.berkeley.edu/gpt-generated-prompts.jsonl -O $OUT_DIR/gpt-generated-prompts.jsonl
wget -q --show-progress http://instruct-pix2pix.eecs.berkeley.edu/human-written-prompts.jsonl -O $OUT_DIR/human-written-prompts.jsonl

# If dataset name isn't provided, exit. 
if [ -z $1 ] 
then 
	exit 0 
fi

# Copy dataset files
mkdir $OUT_DIR/$1
wget -A zip,json -R "index.html*" -q --show-progress -r --no-parent http://instruct-pix2pix.eecs.berkeley.edu/$1/ -nd -P $OUT_DIR/$1/

# Unzip to folders
unzip $OUT_DIR/$1/\*.zip -d $OUT_DIR/$1/

# Cleanup
rm -f $OUT_DIR/$1/*.zip
rm -f $OUT_DIR/$1/*.html
