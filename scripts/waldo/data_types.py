# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0


""" TODO
"""


def validate_image_with_mask(x, num_classes = -1):
    """This function validates an object x that is supposed to represent an image
    with the corresponding object mask.  It returns None if successful;
    otherwise it raises an exception.  Specifically it is checking that:
      x['img'] is a numpy array
      x['mask'] is an integer numpy array of the same size as x['img'] containing
             integer object-ids from 0 ... num-objects - 1.
      x['class'] is a list indexed by object-id giving the class of each object;
            classes should be integers and nonnegative, and if 'num_classes' is
            a value > 0, then the classes should not be >= num_classes."""
    # TODO.
    return None;


def validate_image_with_objects(x):
    """This function validates an object x that is supposed to represent an image
    with a list of objects inside it.  It returns None if successful;
    otherwise it raises an exception.  Specifically it is checking that:
      x['img'] is a numpy array
      x['objects'] is a list of things y satisfying validate_object(y).
    """
    # TODO.
    return None;


def validate_object(x):
    """This function validates an object x that is supposed to represent an object
    inside an image.  It returns None if successful;
    otherwise it raises an exception.  Specifically it is checking that:
      x['polygon'] is a list of >= 3 integer (x,y) pairs representing the corners
                    of the polygon in clockwise or anticlockwise order.
    (We may add further checks later on).
    """
    # TODO.
    return None;


def validate_polygon(x):
    """This function validates an object x that is supposed to represent a polygon.
    This should be a list of >= 3 integer (x,y) pairs representing the corners
    of a polygon in clockwise or anticlockwise order.
    It returns None if successful, otherwise raises an exception."""
    # TODO.
    return None;
