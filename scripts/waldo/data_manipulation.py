# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0


""" TODO
"""


def convert_to_mask(x):
    """ This function accepts an object x that should represent an image
        with polygon objects in it, and returns an object representing an image
        with an object mask.
     """
    validate_image_with_objects(x)

    # TODO ...

    validate_image_with_mask(y)
    return y


def convert_polygon_to_points(polygon):
    """  This function accepts an object representing a polygon as a list of
       points in clockwise or anticlockwise order, and returns the list of
       all the points that are inside that polygon.
    """
    validate_polygon(polygon)

    # TODO ...



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
    # TODO.. set y.
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
    # TODO
    validate_combined_image(y, c)
    return y
