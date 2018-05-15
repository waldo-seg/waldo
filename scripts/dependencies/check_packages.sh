#!/usr/bin/env python3

""" This module will be used to check whether the common requirements 
    that we think the bulk of the example directories will need exist or not.
  It will check the requirements from requirements file 
  (scripts/dependencies/requirements.txt). It will prints the required 
  dependencies that are not present and will exist with status 1 if all dependencies
  are not present.
"""

import pkg_resources
import argparse
import sys
import os
from pkg_resources import DistributionNotFound, VersionConflict

parser = argparse.ArgumentParser(description="Checks if all required packages"
                                "are installed or not. Prints remaining required" 
                                " dependencies and exists with status 1, if all "
                                "packages are not present.",
                                 epilog="E.g.  " + sys.argv[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--requirements_path', type=str, 
                    default="scripts/dependencies/requirements.txt",
                    help='Path of the downloaded requirements file')
args = parser.parse_args()

dependency_list = []
with open(args.requirements_path) as f:
    for line in f:
        line = line.strip()
        # check if required package is installed
        try:
            pkg_resources.require(line)
        except:
            dependency_list.append(line)
    if len(dependency_list) ==0:
        print("{0} all requirements are met".format(sys.argv[0]))
        sys.exit()
    else:
        dependencies_str = " ".join(dependency_list)
        print("{0} not all the required python packages are installed."
              "Please do as follows (from your experiment directory, "
              "e.g. egs/dbs2018/v1): ".format(sys.argv[0]))
        print(". ./cmd.sh \n"
              "pip install {0} ".format(dependencies_str))
        os._exit(1)

