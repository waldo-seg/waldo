#!/usr/bin/env bash

# this script checks dependencies for the ICDAR 2015 dataset

[ -f ./path.sh ] && . ./path.sh
set +e

scripts/dependencies/check_python3.sh
scripts/dependencies/check_numpy.sh
scripts/dependencies/check_pillow.sh
