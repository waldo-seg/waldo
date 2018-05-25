#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

""" This script prepares the training, validation and test data for MADCAT Arabic in a pytorch fashion
"""

import sys
import argparse
import os
import torch
from create_mask_from_page_image import get_mask_from_page_image
from waldo.data_io import DataSaver

parser = argparse.ArgumentParser(description="Creates line images from page image",
                                 epilog="E.g.  " + sys.argv[0] + "  data/LDC2012T15"
                                 " data/LDC2013T09 data/LDC2013T15 data/madcat.train.raw.lineid "
                                 " data/local/lines ",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('database_path1', type=str,
                    help='Path to the downloaded madcat data directory 1')
parser.add_argument('database_path2', type=str,
                    help='Path to the downloaded madcat data directory 2')
parser.add_argument('database_path3', type=str,
                    help='Path to the downloaded madcat data directory 3')
parser.add_argument('data_splits', type=str,
                    help='Path to file that contains the train/test/dev split information')
parser.add_argument('out_dir', type=str,
                    help='directory location to write output files')
parser.add_argument('writing_condition1', type=str,
                    help='Path to the downloaded (and extracted) writing conditions file 1')
parser.add_argument('writing_condition2', type=str,
                    help='Path to the downloaded (and extracted) writing conditions file 2')
parser.add_argument('writing_condition3', type=str,
                    help='Path to the downloaded (and extracted) writing conditions file 3')
parser.add_argument('--max-image-size', type=int, default=256,
                    help='scales down an image if the length of its largest'
                         ' side is greater than max_size')
args = parser.parse_args()


def check_file_location(base_name, wc_dict1, wc_dict2, wc_dict3):
    """ Returns the complete path of the page image and corresponding
        xml file.
    Returns
    -------
    image_file_name (string): complete path and name of the page image.
    madcat_file_path (string): complete path and name of the madcat xml file
                               corresponding to the page image.
    """
    madcat_file_path1 = os.path.join(args.database_path1, 'madcat', base_name + '.madcat.xml')
    madcat_file_path2 = os.path.join(args.database_path2, 'madcat', base_name + '.madcat.xml')
    madcat_file_path3 = os.path.join(args.database_path3, 'madcat', base_name + '.madcat.xml')

    image_file_path1 = os.path.join(args.database_path1, 'images', base_name + '.tif')
    image_file_path2 = os.path.join(args.database_path2, 'images', base_name + '.tif')
    image_file_path3 = os.path.join(args.database_path3, 'images', base_name + '.tif')

    if os.path.exists(madcat_file_path1):
        return madcat_file_path1, image_file_path1, wc_dict1

    if os.path.exists(madcat_file_path2):
        return madcat_file_path2, image_file_path2, wc_dict2

    if os.path.exists(madcat_file_path3):
        return madcat_file_path3, image_file_path3, wc_dict3

    return None, None, None


def parse_writing_conditions(writing_conditions):
    """ Given writing condition file path, returns a dictionary which have writing condition
        of each page image.
    Returns
    ------
    (dict): dictionary with key as page image name and value as writing condition.
    """
    with open(writing_conditions) as f:
        file_writing_cond = dict()
        for line in f:
            line_list = line.strip().split("\t")
            file_writing_cond[line_list[0]] = line_list[3]
    return file_writing_cond


def check_writing_condition(wc_dict, base_name):
    """ Given writing condition dictionary, checks if a page image is writing
        in a specifed writing condition.
        It is used to create subset of dataset based on writing condition.
    Returns
    (bool): True if writing condition matches.
    """
    return True
    writing_condition = wc_dict[base_name].strip()
    if writing_condition != 'IUC':
        return False

    return True


def main():

    wc_dict1 = parse_writing_conditions(args.writing_condition1)
    wc_dict2 = parse_writing_conditions(args.writing_condition2)
    wc_dict3 = parse_writing_conditions(args.writing_condition3)

    splits_handle = open(args.data_splits, 'r')
    splits_data = splits_handle.read().strip().split('\n')

    prev_base_name = ''
    data_saver = DataSaver(args.out_dir)
    for line in splits_data:
        base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
        if prev_base_name != base_name:
            prev_base_name = base_name
            madcat_file_path, image_file_path, wc_dict = check_file_location(base_name, wc_dict1, wc_dict2, wc_dict3)
            if wc_dict is None or not check_writing_condition(wc_dict, base_name):
                continue
            if madcat_file_path is not None:
                y = get_mask_from_page_image(madcat_file_path, image_file_path, args.max_image_size)
                data_saver.write_image(base_name, y)
    data_saver.write_index()

if __name__ == '__main__':
    main()
