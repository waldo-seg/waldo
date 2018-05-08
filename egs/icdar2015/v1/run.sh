#!/usr/bin/env bash


set -e

nj=4

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./scripts/parse_options.sh  # e.g. this parses the above options
                              # if supplied.


./local/check_dependencies.sh
