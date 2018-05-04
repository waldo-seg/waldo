#!/bin/bash

# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

# This script loads the training and test data for DSB2018.

[ -f ./path.sh ] && . ./path.sh; # source the path.

dl_dir=data/download
train_dir=$dl_dir/stage1_train
train_url=http://s3.amazonaws.com/kaggle-dsb2018/stage1_train.zip

train_label_csv=$dl_dir/stage1_train_labels.csv
train_label_url=https://s3.amazonaws.com/kaggle-dsb2018/stage1_train_labels.csv.zip

test_dir=$dl_dir/stage1_test
test_url=https://s3.amazonaws.com/kaggle-dsb2018/stage1_test.zip

mkdir -p $dl_dir
if [ -d $train_dir ]; then
  echo Not downloading DSB2018 training data as it is already there.
else
  if [ ! -f $dl_dir/stage1_train.zip ]; then
    echo Downloading DSB2018 training data...
    wget -P $dl_dir $train_url || exit 1;
  fi
  unzip -qq $dl_dir/stage1_train.zip -d $train_dir || exit 1;
  echo Done downloading and extracting DSB2018 training data
fi


if [ -f $train_label_csv ]; then
  echo Not downloading DSB2018 training label csv file as it is already there.
else
  if [ ! -f $dl_dir/stage1_train_labels.csv.zip ]; then
    echo Downloading DSB2018 training label csv...
    wget -P $dl_dir $train_label_url || exit 1;
  fi
  unzip -qq $dl_dir/stage1_train_labels.csv.zip -d $dl_dir || exit 1;
  echo Done downloading and extracting DSB2018 training label csv file
fi


if [ -d $test_dir ]; then
  echo Not downloading DSB2018 test data as it is already there.
else
  if [ ! -f $dl_dir/stage1_test.zip ]; then
    echo Downloading DSB2018 test data...
    wget -P $dl_dir $test_url || exit 1;
  fi
  unzip -qq $dl_dir/stage1_test.zip -d $test_dir || exit 1;
  echo Done downloading and extracting DSB2018 test data
fi

### Process data and save it to pytorch path file
train_prop=0.9
seed=0
. ./utils/parse_options.sh

mkdir -p data/train_val/split${train_prop}_seed${seed}
mkdir -p data/test

local/process_data.py --train-input $train_dir --test-input $test_dir --outdir data --train-prop $train_prop --img-channels 3 --seed $seed
