# Copyright      2018  Johns Hopkins University (author: Daniel Povey)


# Apache 2.0
from waldo.data_types import *
from PIL import Image
import numpy as np


def randomly_crop_combined_image(combined_image, config,
                                 image_height, image_width):
    """
    This function randomly crops a 'combined image' combined_image
    as would be validated by validate_combined_image(), to an image size
    of 'image_height' by 'image_width'.  It zero-pads if the current
    image is smaller than that.
    It returns the randomly cropped image; it doesn't modify
    'combined_image'.
    """
    validate_combined_image(combined_image, config)

    n_channels, height, width = combined_image.shape

    # it has been made a square image and we only consider one side
    if height <= image_height:
        cropped_image = np.pad(
            combined_image, ((0, 0), (0, image_height - height), (0, image_width - width)), 'constant')
    else:
        top = np.random.randint(0, height - image_height)
        left = np.random.randint(0, width - image_width)
        cropped_image = combined_image[:, top:top +
                                       image_height, left:left + image_width]

    validate_combined_image(cropped_image, config)

    return cropped_image


def scale_down_image_with_objects(image_with_objects, config, max_size):
    """
    This function scales down an image with objects (as would be validated by
    validate_image_with_objects(), if the length of its largest side is
    greater than 'max_size'.  (Otherwise it leaves it the same size).
    It returns the scaled-down image and object; but note, if it does not have
    to scale down the image, it just returns the input variable
    'image_with_objects', it does not make a deep copy.
    """
    validate_image_with_objects(image_with_objects, config)
    im_arr = image_with_objects['img']
    im_max_side = max(im_arr.shape[0], im_arr.shape[1])
    if im_max_side < max_size:
        return image_with_objects

    im = Image.fromarray(im_arr)
    sx = float(im.size[0])
    sy = float(im.size[1])
    scale = 0
    if sy > sx:
        ny = max_size
        scale = (1.0 * ny) / sy
        nx = scale * sx
    else:
        nx = max_size
        scale = (1.0 * nx) / sx
        ny = scale * sy

    resized_img = im.resize((int(nx), int(ny)))
    resized_img_arr = np.array(resized_img)

    resized_objects = []
    for original_object in image_with_objects['objects']:
        ordered_polygon_points = original_object['polygon']
        object = {}
        resized_pp = []
        for point in ordered_polygon_points:
            x, y = point
            new_x = int(x * scale)
            new_y = int(y * scale)
            new_point = (new_x, new_y)
            resized_pp.append(new_point)
        object['polygon'] = resized_pp
        resized_objects.append(object)

    resized_image_with_objects = {
        'img': resized_img_arr,
        'objects': resized_objects
    }

    validate_image_with_objects(resized_image_with_objects, config)

    return resized_image_with_objects


def make_square_image_with_padding(im_arr, num_colors):
    """
    This function pads an image to make it squre, if both height and width are
    different, (Otherwise it leaves it the same size).
    It returns the padded image; but note, if it does not have
    to pad the image, it just returns the input variable
    'image', it does not make a deep copy. Note: it only pads on the right or bottom side.
    """

    assert num_colors == 1 or num_colors == 3
    height = int(im_arr.shape[0])
    width = int(im_arr.shape[1])

    if width == height:
        return im_arr

    if width > height:
        diff = width - height
        if num_colors == 1:
            im_arr_pad = np.pad(
                im_arr, [(0, diff), (0, 0)], mode='constant')
        else:
            im_arr_pad = np.pad(
                im_arr, [(0, diff), (0, 0), (0, 0)], mode='constant')
    else:
        diff = height - width
        if num_colors == 1:
            im_arr_pad = np.pad(
                im_arr, [(0, 0), (0, diff)], mode='constant')
        else:
            im_arr_pad = np.pad(
                im_arr, [(0, 0), (0, diff), (0, 0)], mode='constant')

    return im_arr_pad
