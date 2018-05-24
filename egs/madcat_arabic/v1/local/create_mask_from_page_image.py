#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

""" This module will be used for creating text localization mask on page image.
 Given the word segmentation (bounding box around a word) for every word, it will
 extract line segmentation. To extract line segmentation, it will take word bounding
 boxes of a line as input, will create a minimum area bounding box that will contain
 all corner points of word bounding boxes. The obtained bounding box (will not necessarily
 be vertically or horizontally aligned).
"""

import xml.dom.minidom as minidom
from waldo.data_manipulation import *
from waldo.core_config import CoreConfig
from waldo.mar_utils import compute_hull
from waldo.data_transformation import scale_down_image_with_objects, 
                                      make_square_image_with_padding


def get_mask_from_page_image(madcat_file_path, image_file_name, max_size):
    """ Given a page image, extracts the page image mask from it.
        Input
        -----
        image_file_name (string): complete path and name of the page image.
        madcat_file_path (string): complete path and name of the madcat xml file
                                      corresponding to the page image.
        """

    objects = _get_bounding_box(madcat_file_path)
    img = Image.open(image_file_name).convert("RGB")
    im_arr = np.array(img)

    config = CoreConfig()
    config.num_colors = 3
    image_with_objects = {
        'img': im_arr,
        'objects': objects
    }

    im_height = im_arr.shape[0]
    im_width = im_arr.shape[1]

    validated_objects = []
    for original_object in image_with_objects['objects']:
        ordered_polygon_points = original_object['polygon']
        object = {}
        resized_pp = []
        for point in ordered_polygon_points:
            new_point = _validate_and_update_point(point, im_width, im_height)
            resized_pp.append(new_point)
        object['polygon'] = resized_pp
        validated_objects.append(object)

    validated_image_with_objects = {
        'img': im_arr,
        'objects': validated_objects
    }

    scaled_image_with_objects = scale_down_image_with_objects(validated_image_with_objects, config,
                                                              max_size)

    img_padded = make_squre_image_with_padding(scaled_image_with_objects['img'], config)

    padded_image_with_objects = {
        'img': img_padded,
        'objects': validated_objects
    }

    y = convert_to_mask(padded_image_with_objects, config)

    return y


def _get_bounding_box(madcat_file_path):
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


def _validate_and_update_point(pt0, im_width, im_height, pt1=(0, 0)):
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
