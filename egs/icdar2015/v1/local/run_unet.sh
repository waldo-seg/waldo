#!/bin/bash
stage=0

num_colors=3
num_classes=2
padding=10
train_image_size=128

depth=5
epochs=10
start_filters=64
batch_size=16
lr=0.01
dir=exp/unet_1a

. ./cmd.sh
. ./path.sh
. ./scripts/parse_options.sh




if [ $stage -le 1 ]; then
  mkdir -p $dir/configs
  echo "$0: Creating core configuration and unet configuration"

  cat <<EOF > $dir/configs/core.config
  num_classes $num_classes
  num_colors $num_colors
  padding $padding
  offsets 1 0  0 1  -2 -1  1 -2 
EOF

  cat <<EOF > $dir/configs/unet.config
  depth $depth
  start_filters $start_filters
  up_mode transpose
  merge_mode concat
EOF
fi


if [ $stage -le 2 ]; then
  echo "$0: Training the network....."
  $cmd --gpu 1 --mem 2G $dir/train.log limit_num_gpus.sh local/train.py \
       --train-dir data \
       --batch-size $batch_size \
       --train-image-size 128 \
       --epochs $epochs \
       --lr $lr \
       --core-config $dir/configs/core.config \
       --unet-config $dir/configs/unet.config \
       $dir
fi
