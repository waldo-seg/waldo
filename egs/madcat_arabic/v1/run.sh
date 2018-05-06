#!/usr/bin/env bash

set -e
stage=0
nj=70
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. parse_options.sh  # e.g. this parses the above options
                      # if supplied.

./local/check_tools.sh

mkdir -p data/{train,test,dev}/data
mkdir -p data/local/{train,test,dev}
mkdir -p data/{train,test,dev}/masks
data_splits_dir=data/download/data_splits

if [ $stage -le 0 ]; then
  echo "$0: Downloading data splits..."
  echo "Date: $(date)."
  local/download_data.sh --data_splits $data_splits_dir
fi

if [ $stage -le 0 ]; then
  for dataset in test dev train; do
    echo "$0: Extracting mask from page image for dataset:  $dataset. "
    echo "Date: $(date)."
    dataset_file=$data_splits_dir/madcat.$dataset.raw.lineid
    local/extract_masks.sh --nj $nj --cmd $cmd --dataset_file $dataset_file \
                           --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
                           --download_dir3 $download_dir3 data/local/$dataset
  done
fi

