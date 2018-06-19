#!/usr/bin/env bash


set -e

nj=4
stage=0

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

epochs=120
depth=5
dir=exp/unet_${depth}_${epochs}_sgd
if [ $stage -le 1 ]; then
  # training
  local/run_unet.sh --dir $dir --epochs $epochs --depth $depth --train_prop $train_prop --seed $seed
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
       --csv sub-icdar2015.csv \
       --job JOB --num-jobs $nj

fi

if [ $stage -le 3 ]; then
  echo "doing evaluation..."
  local/scoring.py \
    --ground-truth data/download/stage1_solution.csv \
    --predict $dir/segment/sub-icdar2015.csv \
    --result $dir/segment/result.txt
fi

