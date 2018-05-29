#!/bin/bash

# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

# This script loads the training and test data for DSB2018.

[ -f ./path.sh ] && . ./path.sh; # source the path.

dl_dir=data/download
train_dir=$dl_dir/stage1_train
train_label_csv=$dl_dir/stage1_train_labels.csv
test1_dir=$dl_dir/stage1_test
test1_solution_csv=$dl_dir/stage1_solution.csv
test2_dir=$dl_dir/stage2_test_final

mkdir -p $dl_dir
if [ -d $train_dir ]; then
  echo Not downloading DSB2018 training data as it is already there.
else
  if [ ! -f $dl_dir/stage1_train.zip ]; then
    kaggle competitions download -c data-science-bowl-2018 -f stage1_train.zip -p $dl_dir
  fi
  unzip -qq $dl_dir/stage1_train.zip -d $train_dir || exit 1;
  echo Done downloading and extracting DSB2018 training data
fi


if [ -f $train_label_csv ]; then
  echo Not downloading DSB2018 training label csv file as it is already there.
else
  if [ ! -f $dl_dir/stage1_train_labels.csv.zip ]; then
    kaggle competitions download -c data-science-bowl-2018 -f stage1_train_labels.csv.zip -p $dl_dir
  fi
  unzip -qq $dl_dir/stage1_train_labels.csv.zip -d $dl_dir || exit 1;
  echo Done downloading and extracting DSB2018 training label csv file
fi


if [ -d $test1_dir ]; then
  echo Not downloading DSB2018 stage1_test data as it is already there.
else
  if [ ! -f $dl_dir/stage1_test.zip ]; then
    kaggle competitions download -c data-science-bowl-2018 -f stage1_test.zip -p $dl_dir
  fi
  unzip -qq $dl_dir/stage1_test.zip -d $test1_dir || exit 1;
  echo Done downloading and extracting DSB2018 stage1_test data
fi

if [ -f $test1_solution_csv ]; then
  echo Not downloading DSB2018 stage1_test solution csv as it is already there.
else
  if [ ! -f $dl_dir/stage1_solution.csv.zip ]; then
    kaggle competitions download -c data-science-bowl-2018 -f stage1_solution.csv.zip -p $dl_dir
  fi
  unzip -qq $dl_dir/stage1_solution.csv.zip -d $dl_dir || exit 1;
  echo Done downloading and extracting DSB2018 stage1_test solution csv file
fi


if [ -d $test2_dir ]; then
  echo Not downloading DSB2018 stage2_test data as it is already there.
else
  if [ ! -f $dl_dir/stage2_test.zip ]; then
    kaggle competitions download -c data-science-bowl-2018 -f stage2_test_final.zip -p $dl_dir
  fi
  unzip -qq $dl_dir/stage2_test_final.zip -d $test2_dir || exit 1;
  echo Done downloading and extracting DSB2018 stage2_test data
fi


### Process data and save it to pytorch path file
train_prop=0.9
seed=0
num_classes=2
num_colors=3
. parse_options.sh

mkdir -p data/train
mkdir -p data/val
mkdir -p data/stage1_test
mkdir -p data/stage2_test_final

cat <<EOF > data/core.config
num_classes $num_classes
num_colors $num_colors
EOF

local/process_data.py --indir $dl_dir --outdir data \
		      --train-prop $train_prop --cfg data/core.config --seed $seed
