# Copyright      2018  Johns Hopkins University (author: Daniel Povey)
#                2018  Desh Raj
#                2018  Ashish Arora

# Apache 2.0

import numpy as np
from PIL import Image, ImageDraw
from math import hypot
from waldo.data_types import *
from waldo.mar_utils import get_mar


def convert_to_mask(x, c):
    """ This function accepts an object x that should represent an image
        with polygon objects in it, and returns an object representing an image
        with an object mask. The elements of x are described as

        x['img']: numpy array representation of the image with shape (height, width, colors).
        x['objects']: a dictionary containing all polygon objects in the image under the
        key 'polygon'. Each polygon object is a list of points.
        x['object_class']: an array mapping object ids to their respective classes.
     """
    validate_image_with_objects(x, c)

    im = x['img']
    object_id = 0
    y = dict()
    y['img'] = im
    mask_img = Image.new('L', (im.shape[1], im.shape[0]), 0)
    mask_img_arr = np.array(mask_img)

    object_class = list()
    object_class.append(0)
    for object in x['objects']:
        ordered_polygon_points = object['polygon']
        object_id += 1
        temp_img = Image.new('L', (im.shape[1], im.shape[0]), 0)
        ImageDraw.Draw(temp_img).polygon(
            ordered_polygon_points, fill=object_id)
        temp_img_arr = np.array(temp_img)
        pixels = np.where(temp_img_arr == object_id,
                          temp_img_arr, mask_img_arr)
        array = np.array(pixels, dtype=np.uint8)
        new_image = Image.fromarray(array)
        mask_img_arr = np.array(new_image)
        object_class.append(1)
    y['mask'] = mask_img_arr

    if 'object_class' in x:
        y['object_class'] = x['object_class']
    else:
        y['object_class'] = object_class

    validate_image_with_mask(y, c)
    return y


def compress_image_with_mask(x, c):
    """ This function accepts an object x that should represent an image
        with mask in it, and returns an object representing a compressed image
        with a compressed mask. Specifically, it transforms the image numpy
        array to dtype uint8, and the mask array to dtype uint8 for fewer
        than 256 objects, or to uint16, otherwise.
    """
    validate_image_with_mask(x, c)

    y = dict()
    y['img'] = x['img'].astype(np.uint8)
    if len(x['object_class']) <= 256:
        y['mask'] = x['mask'].astype(np.uint8)
    else:
        y['mask'] = x['mask'].astype(np.uint16)
    y['object_class'] = x['object_class']

    validate_compressed_image_with_mask(y, c)

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

    validate_polygon(points_list)

    return points_list


def get_minimum_bounding_box(polygon):
    """ Given a list of points, returns a minimum area rectangle that will
    contain all points. It will not necessarily be vertically or horizontally
     aligned.
    Returns
    -------
    list((int, int)): 4 corner points of rectangle.
    """
    validate_polygon(polygon)

    points_list = get_mar(polygon)

    validate_polygon(points_list)

    return points_list


def convert_to_combined_image(x, c):
    """ This function processes an 'image-with-mask' x into a 'combined' image,
    containing both input and supervision information in a single numpy array.
    see 'validate_combined_image' in data_types.py for a description of what
    a combined image is.

    This function returns the 'combined' image; it does not modify x.

    The width of the resulting image will be the same as the image in x:
    this function doesn't do padding, you need to call pad_combined_image.
    """
    validate_config(c)
    validate_image_with_mask(x, c)
    # x['img'] is of size (height, width, color), switch it to (color, height, width)
    im = np.moveaxis(x['img'], -1, 0)
    im = im.astype('float32') / 256.0
    mask = x['mask']
    _, height, width = im.shape
    object_class = x['object_class']
    num_outputs = c.num_classes + len(c.offsets)
    num_all_features = c.num_colors + 2 * num_outputs
    y = np.ndarray(
        shape=(num_all_features, height, width), dtype='float32')

    y[:c.num_colors, :, :] = im

    # map object_id to class_id
    def obj_to_class(x):
        return object_class[x]
    class_mask = np.array([[obj_to_class(pixel)
                            for pixel in row] for row in mask])
    for n in range(c.num_classes):
        class_feature = (class_mask == n).astype('float32')
        y[c.num_colors + n, :, :] = class_feature
        y[c.num_colors + num_outputs + n, :, :] = 1 - class_feature

    for k, (i, j) in enumerate(c.offsets):
        rolled_mask = np.roll(np.roll(mask, -i, axis=0), -j, axis=1)
        offset_feature = (rolled_mask == mask).astype('float32')
        y[c.num_colors + c.num_classes + k] = offset_feature
        y[c.num_colors + num_outputs + c.num_classes + k] = 1 - offset_feature

    validate_combined_image(y, c)
    return y


def sort_object_list(objects):
    """Given a list of objects as defined in data_types, returns a new list sorted
    in descending order by the breadth (shorter side) of the rectangles.
    """

    def _get_shorter_side(object):
        """Given an object, returns the length of the shorter side of the associated rectangle
        as a float.
        """
        return min(
            _Euclidean_distance(object['polygon'][0], object['polygon'][1]),
            _Euclidean_distance(object['polygon'][1], object['polygon'][2])
        )

    def _Euclidean_distance(a, b):
        """Given two points, returns their Euclidean distance.
        """
        return hypot(a[0] - b[0], a[1] - b[1])

    map(lambda x: validate_object(x), objects)

    sorted_objects = sorted(objects,
                            key=lambda object: _get_shorter_side(object), reverse=True)
    return sorted_objects


def get_object_class(x):
    """Given a list of objects as defined in the data_types, it returns an array mapping
    object ids to their respective classes.
    """
    # TODO
