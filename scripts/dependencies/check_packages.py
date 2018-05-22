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


def gen_suggestion_str(dependency_list):
    suggestion = ""
    # this list contains packages which require special care in installation
    dependency_list_special = []
    dependency_list_normal = []

    # pick out packages that require special handling
    for pkg in dependency_list:
        if pkg == "torch" or "torch>" in pkg or "torch=" in pkg:
            dependency_list_special.append("torch")
        elif pkg == "torchvision" or "torchvision>" in pkg or "torchvision=" in pkg: 
            dependency_list_special.append("torchvision")
        else:
            dependency_list_normal.append(pkg)

    # generate package installation suggestions
    # [note] default packages installation location:
    # - user mode: ~/.local/lib/pythonX.Y/site-packages
    # - root mode: /usr/local/lib/pythonX.Y/site-packages
    if len(dependency_list_normal) > 0:
        suggestion += "pip3 install --user {0} \n".format(" ".join(dependency_list_normal))
    for pkg in dependency_list_special:
        if pkg == "torch":
            suggestion += "pip3 install --user http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl \n"
        if pkg == "torchvision":
            suggestion += "pip3 install --user torchvision \n"

    return suggestion


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    dependency_list = []
    with open(args.requirements_path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            # check if required package is installed
            try:
                pkg_resources.require(line)
            except:
                dependency_list.append(line)
        if len(dependency_list) == 0:
            print("{0}: All requirements are met".format(sys.argv[0]))
            sys.exit()
        else:
            dependencies_str = " ".join(dependency_list)
            print("{0}: Not all the required python packages are installed. \n".format(sys.argv[0])
                + "Packages required: {0} \n".format(dependencies_str)
                + "Please do as follows "
                + "(from your experiment directory, e.g. egs/dbs2018/v1): \n")

            print(gen_suggestion_str(dependency_list))
            
            print("If you have root access and would like to install " 
                + "system-wide packages, just remove the \"--user\" option.")
            
            os._exit(1)

