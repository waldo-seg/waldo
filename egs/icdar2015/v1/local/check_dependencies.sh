#!/usr/bin/env bash

# exit on error
set -e

# checks if python3 is installed
scripts/dependencies/check_python3.sh

# checks if all required packages are installed or not
scripts/dependencies/check_packages.py

