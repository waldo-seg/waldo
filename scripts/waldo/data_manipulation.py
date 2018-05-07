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
