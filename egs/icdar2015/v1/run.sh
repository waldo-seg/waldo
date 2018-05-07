#!/usr/bin/env bash


set -e
stage=2
nj=4
# download_dir=/export/b18/draj/icdar_2015/
download_dir=/home/desh/Research/icdar/icdar_2015

# for file in $download_dir/*.zip; do
#   unzip $file -d $download_dir && rm $file -d $download_dir
# done

train_images_dir="$download_dir"/ch4_training_images
test_images_dir="$download_dir"/ch4_test_images
train_labels_dir="$download_dir"/ch4_training_labels
test_labels_dir="$download_dir"/ch4_test_labels

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./scripts/parse_options.sh  # e.g. this parses the above options
                              # if supplied.
./local/check_tools.sh

mkdir -p data/lines/{train,test}

if [ $stage -le 0 ]; then
  for dataset in test train; do
    echo "$0: Extracting line images from page image for dataset:  $dataset. "
    echo "Date: $(date)."
    images_dir=$( eval "echo \$${dataset}_images_dir" )
    labels_dir=$( eval "echo \$${dataset}_labels_dir" )
    local/extract_lines.sh --nj $nj --cmd $cmd --images $images_dir \
                           --labels $labels_dir --save_dir data/lines/$dataset
  done
fi

mkdir -p data/masks/{train,test}

if [ $stage -ge 1 ]; then
  for dataset in test train; do
    echo "$0: Extracting mask from page image for dataset:  $dataset. "
    echo "Date: $(date)."
    images_dir=$( eval "echo \$${dataset}_images_dir" )
    labels_dir=$( eval "echo \$${dataset}_labels_dir" )
    local/extract_masks.sh --images $images_dir --labels $labels_dir --save_dir data/masks/$dataset
  done
fi
