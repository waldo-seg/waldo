#!/bin/bash

# Copyright 2018 Johns Hopkins University (author: Desh Raj)
# Apache 2.0

# This script loads the training and test data for ICDAR2015.

[ -f ./path.sh ] && . ./path.sh; # source the path.

dl_dir=${3:-/export/b18/draj/icdar_2015}


if [ ! -d $dl_dir ] ; then
  echo "Please download ICDAR2015 dataset (and labels) and extract in $dl_dir to proceed."
  echo "The extracted directory structure should look like:"
  echo "root"
  echo -e "- train \n -- images \n -- labels"
  echo -e "- test \n -- images \n -- labels"
fi


### Process data and save it to pytorch path file
train_prop=0.9
seed=0
num_classes=2
num_colors=3
. parse_options.sh

mkdir -p data/train
mkdir -p data/val
mkdir -p data/test

cat <<EOF > data/core.config
num_classes $num_classes
num_colors $num_colors
EOF

local/process_data.py --dl_dir $dl_dir --outdir data \
		      --train_prop $train_prop --cfg data/core.config --seed $seed


##Zip test image ground truth labels and save in data directory
zip -j data/test/ground_truth.zip ${dl_dir}/test/labels/*
