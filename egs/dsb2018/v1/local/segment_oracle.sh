#!/bin/bash

dir=exp/oracle
num_classes=2
num_colors=3

mkdir -p $dir/configs
echo "$0: Creating core configuration"

cat <<EOF > $dir/configs/core.config
  num_classes $num_classes
  num_colors $num_colors
  offsets 1 0  0 1  -2 -1  1 -2 
EOF

segdir=$dir/segment

local/segment_oracle.py \
  --train-image-size 1024 \
  --test-data /export/b07/yshao/waldo/egs/dsb2018/v1/data/train \
  --dir $segdir \
