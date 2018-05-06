#!/bin/bash

. ./path.sh

stage=0

# train/validate split
train_prop=0.9
seed=0

# network training setting
gpu_id=0
depth=5
epochs=10
height=128
width=128
batch=16

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. parse_options.sh  # e.g. this parses the --stage option if supplied.


if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --train_prop $train_prop --seed $seed
fi

name=unet_${depth}_${epochs}_sgd
if [ $stage -le 1 ]; then
  # training the network
  $cuda_cmd limit_num_gpus.sh ./local/training.py \
	    --name $name \
	    --depth $depth \
	    --batch-size $batch \
	    --img-height $height \
	    --img-width $width \
	    --epochs $epochs
fi
  
    
    
