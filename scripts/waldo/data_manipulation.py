# Copyright      2018  Johns Hopkins University (author: Daniel Povey)
#                2018  Desh Raj
#                2018  Ashish Arora

# Apache 2.0

import numpy as np
from PIL import Image, ImageDraw
from data_types import *
from data_types import validate_config, validate_image_with_mask, validate_image_with_objects, validate_object, validate_polygon, validate_combined_image


def convert_to_mask(x):
    """ This function accepts an object x that should represent an image
        with polygon objects in it, and returns an object representing an image
        with an object mask.
     """
    validate_image_with_objects(x)

    im = x['img']
    object_id = 0
    y = dict()
    y['img'] = im
    mask_img = Image.new('L', (im.size[0], im.size[1]), 0)
    mask_img_arr = np.array(mask_img)
    object_class = list()
    object_class.append(object_id)
    for object in x['objects']:
        ordered_polygon_points = object['polygon']
        object_id += 1
        temp_img = Image.new('L', (im.size[0], im.size[1]), 0)
        ImageDraw.Draw(temp_img).polygon(ordered_polygon_points, fill=object_id)
        temp_img_arr = np.array(temp_img)
        pixels = np.where(temp_img_arr == object_id, temp_img_arr, mask_img_arr)
        array = np.array(pixels, dtype=np.uint8)
        new_image = Image.fromarray(array)
        mask_img_arr = np.array(new_image)
        object_class.append(object_id)
    y['mask'] = mask_img_arr

    validate_image_with_mask(y)
    return y


def convert_polygon_to_points(polygon):
    """  This function accepts an object representing a polygon as a list of
       points in clockwise or anticlockwise order, and returns the list of
       all the points that are inside that polygon.
    """
    validate_polygon(polygon)

    ordered_polygon_points = polygon['polygon']
    x_list = [i[0] for i in ordered_polygon_points]
    y_list = [i[1] for i in ordered_polygon_points]
    max_x = max(x_list)
    max_y = max(y_list)
    mask_image = Image.new('L', (max_x, max_y), 0)
    ImageDraw.Draw(mask_image).polygon(ordered_polygon_points, fill=1)
    mask_img_arr = np.array(mask_image)
    points_location = np.where(mask_img_arr == 1)
    points_list = []
    for x, y in zip(points_location[0], points_location[1]):
        points_list.append((x, y))

    return points_list


def convert_to_combined_image(x, c):
    """ This function turns an 'image-with-mask' x into a 'combined' image,
    containing both input and supervision information in a single numpy array.
    see 'validate_combined_image' in data_types.py for a description of what
    a combined image is.

    The width of the resulting image will be the same as the image in x:
    this function doesn't do padding, you need to call pad_combined_image.
    """
    validate_config(c)
    validate_image_with_mask(x)
    y = dict()
    # # TODO.. set y.
    validate_combined_image(y, c)
    return y


def pad_combined_image(x, c):
    """ This function adds zero-padding (on both the input and the supervision
    information) to a 'combined image'.  It pads by adding c['padding'] zero pixels
    on each side of x, plus any additional zero-valued pixels as required to
    make the width and height of x at least c['train_image_size'].
    x is not modified; the padded image is returned.
    """
    validate_combined_image(x, c)
    y = dict()
    if c['padding'] == 0:
        return None

    # TODO. set y.
    validate_combined_image(y, c)
    return y


def randomly_crop_combined_image(x, c):
    """ This function randomly crops the combined image 'x' to the size
    c['train_image_size'] by c['train_image_size'], and returns the
    cropped image (x is not modified).  You should probably call
    pad_combined_image before calling this function.

    It is an error if the width or height of image x were previously smaller
    than that. """
    validate_combined_image(x, c)
    y = dict()
    # TODO
    validate_combined_image(y, c)
    return y
