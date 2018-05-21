#!/usr/bin/env bash

# this script checks dependencies for the ICDAR 2015 dataset

# exit on error
set -e

# checks if python3 is installed
scripts/dependencies/check_python3.sh

# checks if all required packages are installed or not
scripts/dependencies/check_packages.sh

