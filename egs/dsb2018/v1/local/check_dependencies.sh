#!/usr/bin/env bash


set -e  # exit on error

scripts/dependencies/check_pytorch.sh

scripts/dependencies/check_packages.sh

