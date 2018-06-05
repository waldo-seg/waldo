#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

. ./scripts/parse_options.sh # e.g. this parses the --stage option if supplied.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

nj=70
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data
writing_condition1=/export/corpora/LDC/LDC2012T15/docs/writing_conditions.tab
writing_condition2=/export/corpora/LDC/LDC2013T09/docs/writing_conditions.tab
writing_condition3=/export/corpora/LDC/LDC2013T15/docs/writing_conditions.tab
local/check_dependencies.sh


if [ $stage -le 0 ]; then
  echo "Preparing data. Date: $(date)."
  local/prepare_data.sh --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
      --download_dir3 $download_dir3 --writing_condition1 $writing_condition1 \
      --writing_condition2 $writing_condition2 --writing_condition3 $writing_condition3
fi


epochs=40
depth=6
lr=0.0005
dir=exp/unet_${depth}_${epochs}_${lr}

if [ $stage -le 1 ]; then
  echo "Training network Date: $(date)."
  local/run_unet.sh --dir $dir --epochs $epochs --depth $depth \
    --train_image_size 256 --lr $lr --batch_size 8
fi

#logdir=$dir/segment/log
#nj=32
#if [ $stage -le 2 ]; then
#  echo "doing segmentation.... Date: $(date)."
#  $cmd JOB=1:$nj $logdir/segment.JOB.log local/segment.py \
#       --train-image-size 256 \
#       --model model_best.pth.tar \
#       --test-data data/test \
#       --dir $dir/segment \
#       --job JOB --num-jobs $nj
#
#fi

if [ $stage -le 2 ]; then
  echo "doing segmentation.... Date: $(date)."
  local/segment.py \
    --train-image-size 256 \
    --model model_best.pth.tar \
    --test-data data/test \
    --dir $dir/segment
fi

if [ $stage -le 3 ]; then
  echo "converting mask to mar format... Date: $(date)."
  for dataset in data/test $dir/segment; do
    scoring/convert_mask_to_mar.py \
      --indir $dataset/mask \
      --outdir $dataset
  done
fi

if [ $stage -le 4 ]; then
  echo "getting score... Date: $(date)."
  scoring/score.py \
    --reference data/test/mar.txt \
    --hypothesis $dir/segment/mar.txt \
    --result $dir/segment/result.txt
fi
echo "Date: $(date)."
