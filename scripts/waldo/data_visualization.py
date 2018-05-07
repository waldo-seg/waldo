# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0


""" TODO
"""


def visualize_mask(x):
    """This function accepts an object x that should represent an image with a
       mask, and it modifies the image to superimpose the "mask" on it.  The
       image will still be visible through a semi-transparent mask layer.
       This function returns None; it modifies x in-place.
    """
    validate_image_with_mask(x)
    # ... do something, modifying x somehow
    return None

def visualize_polygons(x):
    """This function accepts an object x that should represent an image with
       polygonal objects and it modifies the image to superimpose the edges of
       the polygon on it.
       This function returns None; it modifies x in-place.
    """
    validate_image_with_objects(x)
    # ... do something, modifying x somehow
    return None
