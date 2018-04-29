#!/usr/bin/env bash


set -e
stage=1
nj=70
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./scripts/parse_options.sh  # e.g. this parses the above options
                              # if supplied.
./local/check_tools.sh

mkdir -p data/{train,test,dev}/data
mkdir -p data/local/{train,test,dev}

if [ $stage -le 0 ]; then
  for dataset in test dev train; do
    echo "$0: Extracting line images from page image for dataset:  $dataset. "
    echo "Date: $(date)."
    dataset_file=/home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.$dataset.raw.lineid
    local/extract_lines.sh --nj $nj --cmd $cmd --dataset_file $dataset_file \
                           --download_dir1 $download_dir1 --download_dir2 $download_dir2 \
                           --download_dir3 $download_dir3 data/local/$dataset
  done
fi

if [ $stage -le 1 ]; then
  local/get_bounding_box_pixels.py
fi


