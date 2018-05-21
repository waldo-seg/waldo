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


def gen_suggestion(dependency_list, config_case=None):
    suggestion = ""
    # this list contains packages which require special care in installation
    dependency_list_special = []
    dependency_list_rest = []
    
    for pkg in dependency_list:
        # check for torch
        if pkg == "torch" or "torch>" in pkg or "torch=" in pkg:
            dependency_list_special.append("torch")
        # check for torchvision
        elif pkg == "torchvision" or "torchvision>" in pkg or "torchvision=" in pkg: 
            dependency_list_special.append("torchvision")
        else:
            dependency_list_rest.append(pkg)

    # package installation suggestions
    # case 1: for users with root access
    if config_case == "root":
        suggestion += ". ./path.sh \n"
        if len(dependency_list_rest) > 0:
            suggestion += "pip3 install {0} \n".format(" ".join(dependency_list_rest))
        for pkg in dependency_list_special:
            if pkg == "torch":
                suggestion += "pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl \n"
            if pkg == "torchvision":
                suggestion += "pip3 install torchvision \n"

    # case 2: for users without root access while enabling option --user
    elif config_case == "user":
        suggestion += ". ./path.sh \n"
        if len(dependency_list_rest) > 0:
            suggestion += "pip3 install --user {0} \n".format(" ".join(dependency_list_rest))
        for pkg in dependency_list_special:
            if pkg == "torch":
                suggestion += "pip3 install --user http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl \n"
            if pkg == "torchvision":
                suggestion += "pip3 install --user torchvision \n"
 
    # case 3: for users who would like to install miniconda
    elif config_case == "conda":
        for pkg in dependency_list:
            if pkg == "tensorboard_logger" or "tensorboard_logger>" in pkg or "tensorboard_logger=" in pkg: 
                dependency_list_special.append("tensorboard_logger")
                dependency_list_rest.remove(pkg)

        suggestion += ". ./path.sh \n"
        suggestion += ". ./scripts/dependencies/miniconda3/install_miniconda3.sh \n"
        suggestion += ". ./scripts/dependencies/miniconda3/path.sh \n"
        if len(dependency_list_rest) > 0:
            suggestion += "conda install {0} \n".format(" ".join(dependency_list_rest))

        pytorch_installed = False
        for pkg in dependency_list_special:
            if pkg == "torch" and not pytorch_installed:
                suggestion += "conda install pytorch torchvision cuda90 -c pytorch \n"
                pytorch_installed = True
            if pkg == "torchvision" and not pytorch_installed:
                suggestion += "conda install pytorch torchvision cuda90 -c pytorch \n"
                pytorch_installed = True
            if pkg == "tensorboard_logger":
                suggestion += "pip install tensorboard_logger \n"

    # branching should not arrive here
    else:
      print("config_case error in {0}".format(sys.argv[0]))
      os._exit(1)

    return suggestion[:-1]


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
            print("{0} all requirements are met".format(sys.argv[0]))
            sys.exit()
        else:
            dependencies_str = " ".join(dependency_list)
            print("{0} not all the required python packages are installed. \n".format(sys.argv[0])
                  + "Packages required: {0}\n".format(dependencies_str)
                  + "According to your situation, please do as follows (from your experiment directory, "
                  + "e.g. egs/dbs2018/v1): ")
            
            # installing packages to "/usr/local/lib/pythonX.Y/site-packages" by default
            print("\n(1) If you have root access and would like to install system-wide packages: \n" \
                + gen_suggestion(dependency_list, "root"))

            # installing packages to "~/.local/lib/pythonX.Y/site-packages" by default
            print("\n(2) (Recommended) If you do not have root access and would like to install packages under the user scheme: \n" \
                + gen_suggestion(dependency_list, "user"))

            # installing packages to "~/miniconda3/" by default
            print("\n(3) If you would like to use miniconda3 as package manager: \n" \
                + gen_suggestion(dependency_list, "conda"))
            
            os._exit(1)

