#!/bin/bash

# Copyright 2018 Johns Hopkins University (author: Desh Raj)
# Apache 2.0

# This script loads the training and test data for ICDAR2015.

[ -f ./path.sh ] && . ./path.sh; # source the path.

dl_dir=${1:-/export/b18/draj/icdar_2015/}


if [ ! -d $dl_dir ] ; then
  echo "Please download ICDAR2015 dataset (and labels) and extract in $dl_dir to proceed."
  echo "The extracted directory structure should look like:"
  echo "root"
  echo -e "- train \n -- images \n -- labels"
  echo -e "- test \n -- images \n -- labels"
fi


### Process data and save it to pytorch path file
. parse_options.sh

outdir=data
mkdir -p $outdir

train_prop=0.9
seed=0

mkdir -p ${outdir}/train_val/split${train_prop}_seed${seed}
mkdir -p ${outdir}/test

python3 local/process_data.py --dl_dir $dl_dir --outdir $outdir
