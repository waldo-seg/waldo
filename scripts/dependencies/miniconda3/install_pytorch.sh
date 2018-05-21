#!/usr/bin/env bash


# this should be a script that installs pytorch.

if [ ! -f path.sh ]; then
  echo "$0: error: no such file ./path.sh.  Make sure you are running this from one of the example"
  echo "   directories, like egs/dsb2018/v1/"
  exit 1
fi

. ./path.sh

if ! which conda | grep miniconda3; then
  echo "$0: expected `which conda` to return a directory with miniconda3 in its path."
  echo " ... Check that miniconda3 is installed and that your path.sh is set up correctly."
  exit 1
fi

conda install pytorch torchvision cuda91 -c pytorch

exit 0;
