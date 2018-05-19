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
from waldo.data_transformation import scale_down_image_with_objects

import torch
import numpy as np
from PIL import Image
import logging
import math

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
parser.add_argument('--max-image-size', type=int, default=512,
                    help='scales down an image if the length of its largest'
                         ' side is greater than max_size')
args = parser.parse_args()


def get_mask_from_page_image(image_file_name, objects, image_fh):
    """ Given a page image, extracts the page image mask from it.
        Input
        -----
        image_file_name (string): complete path and name of the page image.
        madcat_file_path (string): complete path and name of the madcat xml file
                                      corresponding to the page image.
        """
    im = Image.open(image_file_name)
    im_arr = np.transpose(np.array(im))

    config = CoreConfig()
    image_with_objects = {
        'img': im_arr,
        'objects': objects
    }

    im_width = im_arr.shape[0]
    im_height = im_arr.shape[1]
    
    validated_objects = []
    for original_object in image_with_objects['objects']:
        ordered_polygon_points = original_object['polygon']
        object = {}
        resized_pp = []
        for point in ordered_polygon_points:
            new_point = validate_and_update_point(point, im_width, im_height)
            resized_pp.append(new_point)
        object['polygon'] = resized_pp
        validated_objects.append(object)

    validated_image_with_objects = {
        'img': im_arr,
        'objects': validated_objects
    }
    
    scaled_image_with_objects = scale_down_image_with_objects(validated_image_with_objects, config,
                                                              args.padding)
    y = convert_to_mask(scaled_image_with_objects, config)
    y_mask_arr = y['mask']
    new_image = Image.fromarray(y_mask_arr)
    set_line_image_data(new_image, image_file_name, image_fh)
    return y


def set_line_image_data(image, image_file_name, image_fh):
    """ Given an image, saves the image.
    """
    base_name = os.path.splitext(os.path.basename(image_file_name))[0]
    image_file_name = base_name + '.png'
    image_path = os.path.join(args.out_dir, image_file_name)
    imgray = image.convert('L')
    imgray.save(image_path)
    image_fh.write(image_path + '\n')


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
        mbb_input = []
        for token_node in token_image:
            word_point = token_node.getElementsByTagName('point')
            for word_node in word_point:
                word_coordinate = (int(word_node.getAttribute('x')), int(word_node.getAttribute('y')))
                mbb_input.append(word_coordinate)
        points = get_minimum_bounding_box(mbb_input)
        points_ordered = compute_hull(points)
        points_ordered = points_ordered[:-1]
        object['polygon'] = points_ordered
        objects.append(object)
    return objects


def validate_and_update_point(pt0, im_width, im_height, pt1=(0, 0)):
    new_point = pt0
    if pt0[0] < 0:
        new_point = _get_pointx_inside_origin(pt0, pt1)

    if pt0[0] > im_width:
        new_point = _get_pointx_inside_width(pt0, pt1, im_width)

    if pt0[1] < 0:
        new_point = _get_pointy_inside_origin(pt0, pt1)

    if pt0[1] > im_height:
        new_point = _get_pointy_inside_height(pt0, pt1, im_height)

    return new_point

def _unit_vector(pt0, pt1):
    """ Given two points pt0 and pt1, return a unit vector that
        points in the direction of pt0 to pt1.
    Returns
    -------
    (float, float): unit vector
    """
    dis_0_to_1 = math.sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def _orthogonal_vector(vector):
    """ Given a vector, returns a orthogonal/perpendicular vector of equal length.
    Returns
    ------
    (float, float): A vector that points in the direction orthogonal to vector.
    """
    return -1 * vector[1], vector[0]


def _get_vector_angle(pt0, pt1):
    """ Given two points pt0 and pt1, return a unit vector angle
    Returns
    -------
    float: angle in radian
    """
    vector = _unit_vector(pt0, pt1)
    radian = math.atan2(vector[1], vector[0])
    return radian


def _get_pointx_inside_origin(pt0, pt1):
    """ Given a point pt0, return an updated point that is
    inside orgin. It finds line equation and uses it to
    get updated point x value inside origin
    Returns
    -------
    (float, float): updated point
    """
    return (0, pt0[1])
    # TODO
    x_0 = pt0[0]
    y_0 = pt0[1]

    slope = math.tan(_get_vector_angle(pt0, pt1))
    new_x = 0
    if 0 <= slope <= 10 or -10 <= slope <= 0:
        new_y = slope * -1 * x_0 + y_0
    else:
        raise Exception("Both x's cannot be too close and outside image")

    new_point = (new_x, new_y)

    return new_point


def _get_pointx_inside_width(pt0, pt1, im_width):
    """ Given a point pt0, return an updated point that is
    inside image width. It finds line equation and uses it to
    get updated point x value inside image width
    Returns
    -------
    (float, float): updated point
    """
    return (im_width, pt0[1])
    # TODO
    x_0 = pt0[0]
    y_0 = pt0[1]

    slope = math.tan(_get_vector_angle(pt0, pt1))
    new_x = im_width
    if 0 <= slope <= 10 or -10 <= slope <= 0:
        new_y = slope * (im_width - x_0) + y_0
    else:
        raise Exception("Both x's cannot be too close outside image")

    new_point = (new_x, new_y)
    return new_point


def _get_pointy_inside_origin(pt0, pt1):
    """ Given a point pt0, return an updated point that is
    inside orgin. It finds line equation and uses it to
    get updated point y value inside origin
    Returns
    -------
    (float, float): updated point
    """
    return (pt0[0], 0)
    # TODO
    x_0 = pt0[0]
    y_0 = pt0[1]

    slope = math.tan(_get_vector_angle(pt0, pt1))
    new_y = 0
    if slope >=0.01 or slope <= -0.01:
        new_x = ((-1 * y_0)/slope) + x_0
    else:
        raise Exception("Both y's cannot be too close outside image")

    new_point = (new_x, new_y)
    return new_point


def _get_pointy_inside_height(pt0, pt1, im_height):
    """ Given a point pt0, return an updated point that is
    inside image height. It finds line equation and uses it to
    get updated point y value inside image height
    Returns
    -------
    (float, float): updated point
    """
    return (pt0[0], im_height)
    # TODO
    x_0 = pt0[0]
    y_0 = pt0[1]

    slope = math.tan(_get_vector_angle(pt0, pt1))
    new_y = im_height
    if slope >= 0.01 or slope <= -0.01:
        new_x = ((im_height - y_0) / slope) + x_0
    else:
        raise Exception("Both y's cannot be too close outside image")

    new_point = (new_x, new_y)

    return new_point


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


