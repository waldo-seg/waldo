#!/usr/bin/env python3

# Copyright   2018 Ashish Arora
#             2018 Desh Raj
# Apache 2.0

""" This module will be used for creating text localization mask on page image.
 Given the word segmentation (bounding box around a word) for every word, it will
 extract line segmentation. To extract line segmentation, it will take word bounding
 boxes of a line as input, will create a minimum area bounding box that will contain
 all corner points of word bounding boxes. The obtained bounding box (will not necessarily
 be vertically or horizontally aligned). To obtain the pixel mask, page image is
 rotated to make the bounding box horizontal. In the horizontal bounding box
 pixel locations in the box are reversed mapped to unrotated image
"""

import sys
import argparse
import os
import xml.dom.minidom as minidom
from waldo.data_manipulation import *
from waldo.core_config import CoreConfig
from waldo.mar_utils import compute_hull
from waldo.data_visualization import visualize_mask

import torch
import numpy as np
from PIL import Image
import logging


sys.path.insert(0, 'steps')
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

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
parser.add_argument('--padding', type=int, default=400,
                    help='padding across horizontal/verticle direction')
parser.add_argument('--downsampling_ratio', type=int, default=2,
                    help='ratio of original image and resized image')
args = parser.parse_args()


def pad_image(image):
    """ Given an image, returns a padded image around the border.
        This routine save the code from crashing if bounding boxes that are
        slightly outside the page boundary.
    Returns
    -------
    image: page image
    """
    offset = int(args.padding // 2)
    padded_image = Image.new('L', (image.size[0] + int(args.padding), image.size[1] + int(args.padding)), "white")
    padded_image.paste(im=image, box=(offset, offset))
    return padded_image


def update_minimum_bounding_box_input(bounding_box_input):
    """ Given list of 2D points, returns list of 2D points shifted by an offset.
    Returns
    ------
    points [(float, float)]: points, a list or tuple of 2D coordinates
    """
    paded_mbb = []
    offset = int(args.padding // 2)
    for point in bounding_box_input:
        x, y = point
        new_x = x + offset
        new_y = y + offset
        new_point = (new_x, new_y)
        paded_mbb.append(new_point)

    resized_mbb = []
    ratio = int(args.downsampling_ratio)
    for point in paded_mbb:
        x, y = point
        new_x = int(x/ratio)
        new_y = int(y/ratio)
        new_point = (new_x, new_y)
        resized_mbb.append(new_point)

    return resized_mbb


def get_mask_from_page_image(image_file_name, objects, image_fh):
    """ Given a page image, extracts the page image mask from it.
    Input
    -----
    image_file_name (string): complete path and name of the page image.
    madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.
    """
    im_wo_pad = Image.open(image_file_name)
    im_pad = pad_image(im_wo_pad)
    im_resized = downsample_image(im_pad)
    im_arr = np.array(im_resized)

    config = CoreConfig()
    config.padding = int(args.padding // 2)
    base_name = os.path.splitext(os.path.basename(image_file_name))[0]
    config_path = os.path.join(args.out_dir, base_name + '.txt')
    config.write(config_path)

    image_with_objects = {
        'img': im_arr,
        'objects': objects
    }

    y = convert_to_mask(image_with_objects, config)
    return y


def downsample_image(image):
    """ Given an image, returns a resized image.
    Returns
    -------
    image: page image
    """
    ratio = int(args.downsampling_ratio)
    sx = float(image.size[0])
    sy = float(image.size[1])
    new_sx = sx/ratio
    new_sy = sy/ratio

    img = image.resize((int(new_sx), int(new_sy)))
    return img


def get_bounding_box(madcat_file_path):
    """ Given word boxes of each line, return bounding box for each
     line in sorted order
    Input
    -----
    image_file_name (string): complete path and name of the page image.
    madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.
    """
    objects = []
    doc = minidom.parse(madcat_file_path)
    zone = doc.getElementsByTagName('zone')
    for node in zone:
        object = {}
        token_image = node.getElementsByTagName('token-image')
        minimum_bounding_box_input = []
        for token_node in token_image:
            word_point = token_node.getElementsByTagName('point')
            for word_node in word_point:
                word_coordinate = (int(word_node.getAttribute('x')), int(word_node.getAttribute('y')))
                minimum_bounding_box_input.append(word_coordinate)
        updated_mbb_input = update_minimum_bounding_box_input(minimum_bounding_box_input)
        points = get_minimum_bounding_box(updated_mbb_input)
        points_ordered = compute_hull(points)
        object['polygon'] = points_ordered
        objects.append(object)
    return objects


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

    writing_condition_folder_list = args.database_path1.split('/')
    writing_condition_folder1 = ('/').join(writing_condition_folder_list[:5])

    writing_condition_folder_list = args.database_path2.split('/')
    writing_condition_folder2 = ('/').join(writing_condition_folder_list[:5])

    writing_condition_folder_list = args.database_path3.split('/')
    writing_condition_folder3 = ('/').join(writing_condition_folder_list[:5])

    writing_conditions1 = os.path.join(writing_condition_folder1, 'docs', 'writing_conditions.tab')
    writing_conditions2 = os.path.join(writing_condition_folder2, 'docs', 'writing_conditions.tab')
    writing_conditions3 = os.path.join(writing_condition_folder3, 'docs', 'writing_conditions.tab')

    wc_dict1 = parse_writing_conditions(writing_conditions1)
    wc_dict2 = parse_writing_conditions(writing_conditions2)
    wc_dict3 = parse_writing_conditions(writing_conditions3)

    splits_handle = open(args.data_splits, 'r')
    splits_data = splits_handle.read().strip().split('\n')
    image_file = os.path.join(args.out_dir, 'images.scp')
    image_fh = open(image_file, 'w', encoding='utf-8')

    data = []
    output_path = args.out_dir + '.pth.tar'
    prev_base_name = ''
    for line in splits_data:
        base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
        if prev_base_name != base_name:
            prev_base_name = base_name
            madcat_file_path, image_file_path, wc_dict = check_file_location(base_name, wc_dict1, wc_dict2, wc_dict3)
            if wc_dict is None or not check_writing_condition(wc_dict, base_name):
                continue
            if madcat_file_path is not None:
                objects = get_bounding_box(madcat_file_path)
                y = get_mask_from_page_image(image_file_path, objects, image_fh)
                y['name'] = base_name
                data.append(y)

    torch.save(data, output_path)

if __name__ == '__main__':
      main()

