#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

# train/validate split
train_prop=0.9
seed=0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. parse_options.sh  # e.g. this parses the --stage option if supplied.


local/check_dependencies.sh

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --train_prop $train_prop --seed $seed
fi
