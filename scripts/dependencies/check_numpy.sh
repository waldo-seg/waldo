#!/bin/bash

# this script checks if numpy is installed

python3 -c "import numpy"
if [ $? -ne 0 ] ; then
  echo >&2 "This recipe needs numpy installed."
  exit 1
fi