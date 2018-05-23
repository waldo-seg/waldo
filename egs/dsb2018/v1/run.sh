#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

. parse_options.sh  # e.g. this parses the --stage option if supplied.


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

local/check_dependencies.sh


# train/validate split
train_prop=0.9
seed=0
if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --train_prop $train_prop --seed $seed
fi


epochs=10
depth=5
dir=exp/unet_${depth}_${epochs}_sgd
if [ $stage -le 1 ]; then
  # training
  local/run_unet.sh --epochs $epochs --depth $depth
fi

if [ $stage -le 2 ]; then
    echo "doing segmentation...."
  local/segment.py \
    --train-image-size 128 \
    --model model_best.pth.tar \
    data/val \
    $dir

fi
