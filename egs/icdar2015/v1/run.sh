#!/usr/bin/env bash


set -e

nj=4
stage=3

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./scripts/parse_options.sh  # e.g. this parses the above options
                              # if supplied.


local/check_dependencies.sh

# train/validate split
train_prop=0.9
seed=0
if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh $train_prop $seed
fi

epochs=50
depth=5
dir=exp/unet_${depth}_${epochs}_sgd
if [ $stage -le 1 ]; then
  # training
  local/run_unet.sh --dir $dir --epochs $epochs --depth $depth
fi

logdir=$dir/segment/log
nj=10
if [ $stage -le 2 ]; then
    echo "doing segmentation...."
  $cmd JOB=1:$nj $logdir/segment.JOB.log local/segment.py \
       --train-image-size 128 \
       --model model_best.pth.tar \
       --test-data data/test \
       --dir $dir/segment \
       --job JOB --num-jobs $nj

fi

#Preparation for scoring
mkdir -p $dir/segment/results
zip -j $dir/segment/lbl.zip $dir/segment/lbl/*

if [ $stage -le 3 ]; then
  echo "doing evaluation..."
  python3 local/eval/script.py \
    -g=data/test/ground_truth.zip \
    -s=$dir/segment/lbl.zip \
    -o=$dir/segment/results
fi

