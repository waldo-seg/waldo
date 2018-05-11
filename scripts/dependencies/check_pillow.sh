#!/bin/bash

# this script checks if Pillow is installed as a module

python3 -c "import PIL"
if [ $? -ne 0 ] ; then
  echo >&2 "This recipe needs PIL (Pillow) installed."
  exit 1
fi