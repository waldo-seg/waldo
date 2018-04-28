#!/usr/bin/env bash


set -e
stage=0

. ./scripts/parse_options.sh  # e.g. this parses the above options
                            # if supplied.

local/get_bounding_box_pixels.py
