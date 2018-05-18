#!/usr/bin/env bash

export PATH="${HOME}/anaconda3/bin:$PATH"
export PYTHONPATH="${PYTHONPATH}:${HOME}/anaconda3"

export PYTHONPATH="${PYTHONPATH}:scripts"
export PATH=$PWD/../../../scripts/parallel/:$PATH
export PATH=$PWD/../../../scripts/waldo:$PATH
export PATH=$PWD/../../../scripts:$PATH
export PYTHONPATH=${PYTHONPATH}:$PWD/../../../scripts
