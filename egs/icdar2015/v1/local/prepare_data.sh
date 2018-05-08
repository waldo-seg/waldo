#!/bin/bash

# Copyright 2018 Johns Hopkins University (author: Desh Raj)
# Apache 2.0

# This script loads the training and test data for ICDAR2015.

[ -f ./path.sh ] && . ./path.sh; # source the path.

dl_dir=/export/b18/draj/icdar_2015

train_images_dir="$download_dir"/train/images
test_images_dir="$download_dir"/test/images
train_labels_dir="$download_dir"/train/labels
test_labels_dir="$download_dir"/test/labels


mkdir -p $train_images_dir
mkdir -p $train_labels_dir
mkdir -p $test_images_dir
mkdir -p $test_labels_dir

if [ ! -d $train_dir ] || [ ! -d $test_dir ] ; then
  echo "Please download ICDAR2015 dataset (and labels) and extract in created directories 
  in $dl_dir to proceed."
fi


### Process data and save it to pytorch path file
. parse_options.sh

# local/process_data.py --train-input $train_dir --test-input $test_dir --outdir data --train-prop $train_prop --img-channels 3 --seed $seed
