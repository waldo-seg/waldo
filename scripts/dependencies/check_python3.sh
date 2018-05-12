#!/bin/bash

# this script checks if Python 3 is installed

command -v python3 >&/dev/null \
|| { echo >&2 "python3 not found on PATH. You will have to install Python3, preferably >= 3.6"; exit 1; }