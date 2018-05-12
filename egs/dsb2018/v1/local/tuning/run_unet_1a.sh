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

. ./cmd.sh
. ./path.sh
. ./scripts/parse_options.sh


dir=exp/unet_${depth}_${epochs}_sgd


if [ $stage -le 1 ]; then
  mkdir -p $dir/configs
  echo "$0: creating core configuration and unet configuration"
  
  cat <<EOF > $dir/configs/core.config
  num_classes $num_classes
  num_colors $num_colors
  padding $padding
  train_image_size ${train_image_size}
  offsets 1 0  0 1  -2 -1  1 -2  3 2  -4 3  -4 -7  10 -4  3 15  -21 0
EOF

  cat <<EOF > $dir/configs/unet.config
  depth $depth
  start_filters $start_filters
  up_mode transpose
  merge_mode concat
EOF
fi


if [ $stage -le 2 ]; then
  # training the network
  $cmd --gpu 1 --mem 2G $dir/train.log limit_num_gpus.sh local/train.py \
       --batch-size $batch_size \
       --epochs $epochs \
       --lr $lr \
       --core-config $dir/configs/core.config \
       --unet-config $dir/configs/unet.config \
       $dir
fi

exit 0;
