#!/usr/bin/env python3

""" This module will be used to check whether the common requirements 
    that we think the bulk of the example directories will need exist or not.
  It will check the requirements from requirements files 
  (scripts/dependencies/requirements.txt, ./requirements.txt and the ones specified).
  It will prints the required dependencies that are not present 
  and will exist with status 1 if all dependencies are not present.
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
                    action='append',
                    # default="scripts/dependencies/requirements.txt",
                    help='Path of the downloaded requirements file')


def gen_suggestion_str(dependencies_list):
    suggestion = ""
    # this list contains packages which require special care in installation
    dependencies_list_special = []
    dependencies_list_normal = []

    # pick out packages that require special handling
    for pkg in dependencies_list:
        if pkg == "torch" or "torch>" in pkg or "torch=" in pkg:
            dependencies_list_special.append("torch")
        elif pkg == "torchvision" or "torchvision>" in pkg or "torchvision=" in pkg: 
            dependencies_list_special.append("torchvision")
        else:
            dependencies_list_normal.append(pkg)

    # generate package installation suggestions
    # [note] default packages installation location:
    # - user mode: ~/.local/lib/pythonX.Y/site-packages
    # - root mode: /usr/local/lib/pythonX.Y/site-packages
    if len(dependencies_list_normal) > 0:
        suggestion += "pip3 install --user {0} \n".format(" ".join(dependencies_list_normal))
    for pkg in dependencies_list_special:
        if pkg == "torch":
            suggestion += "pip3 install --user http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl \n"
        if pkg == "torchvision":
            suggestion += "pip3 install --user torchvision \n"

    return suggestion


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    requirements_paths = []
    requirements_paths.append("scripts/dependencies/requirements.txt") # global requirements
    requirements_paths.append("requirements.txt") # local requirements
    if args.requirements_path is not None:
        requirements_paths += args.requirements_path
    requirements_paths = list(set(requirements_paths))

    dependencies_list = []
    for path in requirements_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("#"):
                    continue
                # check if required package is installed
                try:
                    pkg_resources.require(line)
                except:
                    dependencies_list.append(line)

    if len(dependencies_list) == 0:
        print("{0}: All requirements are met".format(sys.argv[0]))
        sys.exit()
    else:
        dependencies_list.sort()
        dependencies_list = list(set(dependencies_list)) # Simple deduplication. No considering same package with different version requirements
        dependencies_str = " ".join(dependencies_list)
        print("{0}: Not all the required python packages are installed. \n".format(sys.argv[0])
            + "Packages required: {0} \n".format(dependencies_str)
            + "Please do as follows "
            + "(from your experiment directory, e.g. egs/dbs2018/v1): \n")

        print(gen_suggestion_str(dependencies_list))
        
        print("If you have root access and would like to install " 
            + "system-wide packages, just remove the \"--user\" option.")
        
        os._exit(1)

