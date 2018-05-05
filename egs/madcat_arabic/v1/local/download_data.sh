#!/bin/bash

# Copyright      2018  Ashish Arora
# Apache 2.0

# This script downloads data splits for MADCAT Arabic dataset.
# It also check if madcat arabic data is present or not.

train_split_url=http://www.openslr.org/resources/48/madcat.train.raw.lineid
test_split_url=http://www.openslr.org/resources/48/madcat.test.raw.lineid
dev_split_url=http://www.openslr.org/resources/48/madcat.dev.raw.lineid
data_splits=data/download/data_splits

. ./cmd.sh
. ./path.sh
. parse_options.sh || exit 1;

if [ -d $data_splits ]; then
  echo "$0: Not downloading the data splits as it is already there."
else
  if [ ! -f $data_splits/madcat.train.raw.lineid ]; then
    mkdir -p $data_splits
    echo "$0: Downloading the data splits..."
    wget -P $data_splits $train_split_url || exit 1;
    wget -P $data_splits $test_split_url || exit 1;
    wget -P $data_splits $dev_split_url || exit 1;
  fi
  echo "$0: Done downloading the data splits"
fi

