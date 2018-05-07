#!/usr/bin/env bash

# The script automatically choose default settings of miniconda for installation
# Miniconda will be installed in the HOME directory. ($HOME/miniconda3).
# Also don't make miniconda's python as default.

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b

# NOTE: if you install this elsewhere than in your home directory, you'll have
# to edit your path.sh file to reflect that change.  Search in that file for
# miniconda3 and you'll see.
