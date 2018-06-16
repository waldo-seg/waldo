#!/usr/bin/env python3

import sys
import argparse
import os
import numpy as np
from math import atan2, cos, sin, pi, degrees
from collections import namedtuple

from PIL import Image
from scipy.misc import toimage
from glob import glob
from waldo.mar_utils import get_rectangle

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
parser.add_argument('mar_file_path', type=str,
                    help='Path to the minimum area rectangle file')
args = parser.parse_args()

"""
bounding_box is a named tuple which contains:
             area (float): area of the rectangle
             length_parallel (float): length of the side that is parallel to unit_vector
             length_orthogonal (float): length of the side that is orthogonal to unit_vector
             rectangle_center(int, int): coordinates of the rectangle center
             (use rectangle_corners to get the corner points of the rectangle)
             unit_vector (float, float): direction of the length_parallel side.
             (it's orthogonal vector can be found with the orthogonal_vector function
             unit_vector_angle (float): angle of the unit vector to be in radians.
             corner_points [(float, float)]: set that contains the corners of the rectangle
"""

bounding_box_tuple = namedtuple('bounding_box_tuple', 'area '
                                        'length_parallel '
                                        'length_orthogonal '
                                        'rectangle_center '
                                        'unit_vector '
                                        'unit_vector_angle '
                                        'corner_points'
                         )


def orthogonal_vector(vector):
    """ Given a vector, returns a orthogonal/perpendicular vector of equal length.
    Returns
    ------
    (float, float): A vector that points in the direction orthogonal to vector.
    """
    return -1 * vector[1], vector[0]


def get_center(im):
    """ Given image, returns the location of center pixel
    Returns
    -------
    (int, int): center of the image
    """
    center_x = im.size[0] / 2
    center_y = im.size[1] / 2
    return int(center_x), int(center_y)


def get_horizontal_angle(unit_vector_angle):
    """ Given an angle in radians, returns angle of the unit vector in
        first or fourth quadrant.
    Returns
    ------
    (float): updated angle of the unit vector to be in radians.
             It is only in first or fourth quadrant.
    """
    if unit_vector_angle > pi / 2 and unit_vector_angle <= pi:
        unit_vector_angle = unit_vector_angle - pi
    elif unit_vector_angle > -pi and unit_vector_angle < -pi / 2:
        unit_vector_angle = unit_vector_angle + pi

    return unit_vector_angle


def get_smaller_angle(bounding_box):
    """ Given a rectangle, returns its smallest absolute angle from horizontal axis.
    Returns
    ------
    (float): smallest angle of the rectangle to be in radians.
    """
    unit_vector = bounding_box.unit_vector
    unit_vector_angle = bounding_box.unit_vector_angle
    ortho_vector = orthogonal_vector(unit_vector)
    ortho_vector_angle = atan2(ortho_vector[1], ortho_vector[0])

    unit_vector_angle_updated = get_horizontal_angle(unit_vector_angle)
    ortho_vector_angle_updated = get_horizontal_angle(ortho_vector_angle)

    if abs(unit_vector_angle_updated) < abs(ortho_vector_angle_updated):
        return unit_vector_angle_updated
    else:
        return ortho_vector_angle_updated


def rotated_points(bounding_box, center):
    """ Given the rectangle, returns corner points of rotated rectangle.
        It rotates the rectangle around the center by its smallest angle.
    Returns
    -------
    [(int, int)]: 4 corner points of rectangle.
    """
    p1, p2, p3, p4 = bounding_box.corner_points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    center_x, center_y = center
    rotation_angle_in_rad = -get_smaller_angle(bounding_box)
    x_dash_1 = (x1 - center_x) * cos(rotation_angle_in_rad) - (y1 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_2 = (x2 - center_x) * cos(rotation_angle_in_rad) - (y2 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_3 = (x3 - center_x) * cos(rotation_angle_in_rad) - (y3 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_4 = (x4 - center_x) * cos(rotation_angle_in_rad) - (y4 - center_y) * sin(rotation_angle_in_rad) + center_x

    y_dash_1 = (y1 - center_y) * cos(rotation_angle_in_rad) + (x1 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_2 = (y2 - center_y) * cos(rotation_angle_in_rad) + (x2 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_3 = (y3 - center_y) * cos(rotation_angle_in_rad) + (x3 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_4 = (y4 - center_y) * cos(rotation_angle_in_rad) + (x4 - center_x) * sin(rotation_angle_in_rad) + center_y
    return x_dash_1, y_dash_1, x_dash_2, y_dash_2, x_dash_3, y_dash_3, x_dash_4, y_dash_4


def set_line_image_data(image, line_id, image_file_name, image_fh):
    """ Given an image, saves a flipped line image. Line image file name
        is formed by appending the line id at the end page image name.
    """

    base_name = os.path.splitext(os.path.basename(image_file_name))[0]
    line_id = '_' + line_id.zfill(4)
    line_image_file_name = base_name + line_id + '.png'
    image_path = os.path.join(args.out_dir, line_image_file_name)
    imgray = image.convert('L')
    imgray_rev_arr = np.fliplr(imgray)
    imgray_rev = toimage(imgray_rev_arr)
    imgray_rev.save(image_path)
    image_fh.write(image_path + '\n')
    
    
def get_line_image_from_mar(image_file_name, image_fh, mar):
    """ Given a page image, extracts the line images from it.
    Input
    -----
    image_file_name (string): complete path and name of the page image.
    madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.
    """

    im = Image.open(image_file_name)
    bounding_box = get_rectangle([(mar[1], mar[0]), (mar[3], mar[2]), (mar[5], mar[4]), (mar[7], mar[6])])
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bounding_box.corner_points
    min_x, min_y = int(min(x1, x2, x3, x4)), int(min(y1, y2, y3, y4))
    max_x, max_y = int(max(x1, x2, x3, x4)), int(max(y1, y2, y3, y4))

    box = (min_x, min_y, max_x, max_y)
    region_initial = im.crop(box)
    rot_points = []
    p1, p2 = (x1 - min_x, y1 - min_y), (x2 - min_x, y2 - min_y)
    p3, p4 = (x3 - min_x, y3 - min_y), (x4 - min_x, y4 - min_y)
    rot_points.append(p1)
    rot_points.append(p2)
    rot_points.append(p3)
    rot_points.append(p4)

    cropped_bounding_box = bounding_box_tuple(bounding_box.area,
                                              bounding_box.length_parallel,
                                              bounding_box.length_orthogonal,
                                              bounding_box.length_orthogonal,
                                              bounding_box.unit_vector,
                                              bounding_box.unit_vector_angle,
                                              set(rot_points)
                                              )

    rotation_angle_in_rad = get_smaller_angle(cropped_bounding_box)
    img2 = region_initial.rotate(degrees(rotation_angle_in_rad), resample=Image.BICUBIC)
    x_dash_1, y_dash_1, x_dash_2, y_dash_2, x_dash_3, y_dash_3, x_dash_4, y_dash_4 = rotated_points(
        cropped_bounding_box, get_center(region_initial))

    min_x = int(min(x_dash_1, x_dash_2, x_dash_3, x_dash_4))
    min_y = int(min(y_dash_1, y_dash_2, y_dash_3, y_dash_4))
    max_x = int(max(x_dash_1, x_dash_2, x_dash_3, x_dash_4))
    max_y = int(max(y_dash_1, y_dash_2, y_dash_3, y_dash_4))
    box = (min_x, min_y, max_x, max_y)
    region_final = img2.crop(box)
    set_line_image_data(region_final, '0', image_file_name, image_fh)


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
        in a specified writing condition.
        It is used to create subset of dataset based on writing condition.
    Returns
    (bool): True if writing condition matches.
    """
    #return True
    writing_condition = wc_dict[base_name].strip()
    if writing_condition != 'IUC':
        return False

    return True


def get_file_list():
    """ Given writing condition and data splits, it returns file list
    that will be processed. writing conditions helps in creating subset
    of dataset from full dataset.
    Returns
    []: list of files to be processed.
    """

    wc_dict1 = parse_writing_conditions(args.writing_condition1)
    wc_dict2 = parse_writing_conditions(args.writing_condition2)
    wc_dict3 = parse_writing_conditions(args.writing_condition3)
    splits_handle = open(args.data_splits, 'r')
    splits_data = splits_handle.read().strip().split('\n')

    file_list = list()
    prev_base_name = ''
    for line in splits_data:
        base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
        if prev_base_name != base_name:
            prev_base_name = base_name
            madcat_file_path, image_file_path, wc_dict = check_file_location(base_name, wc_dict1, wc_dict2, wc_dict3)
            if wc_dict is None or not check_writing_condition(wc_dict, base_name):
                continue
            file_list.append((madcat_file_path, image_file_path, base_name))

    return file_list


def read_rect_coordinates(mar_file_path):
    image_rect_dict = dict()
    with open(mar_file_path) as f:
        for line in f:
            line_vect = line.strip().split(' ')
            image_id = line_vect[0]
            if image_id not in image_rect_dict.keys():
                image_rect_dict[image_id] = dict()
            line_id = line_vect[1]
            hyp_rect = line_vect[2]
            image_rect_dict[image_id][line_id] = hyp_rect

    return image_rect_dict


### main ###
def main():
    file_list = get_file_list()
    image_file = os.path.join(args.out_dir, 'images.scp')
    image_fh = open(image_file, 'w', encoding='utf-8')
    image_rect_dict = read_rect_coordinates(args.mar_file_path)
    for file_name in file_list:
        image_id = os.path.splitext(os.path.basename(file_name[1]))[0]
        for line_id in image_rect_dict[image_id]:
            mar = image_rect_dict[image_id][line_id]
            #print(mar)
            get_line_image_from_mar(file_name[1], image_fh, mar)

if __name__ == '__main__':
      main()

