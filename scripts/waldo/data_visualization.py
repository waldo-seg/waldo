# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0


""" TODO
"""

def visualize_mask(x):
    # "image" is a dictionary containing masks and image and numpy arrays
    mask = x['mask']
    red, green, blue, alpha = mask.T  # unpacking blends
    white_areas = (red > 0) & (blue > 0) & (green > 0)
    black_areas = (red == 0) & (blue == 0) & (green == 0) & (alpha == 255)
    # randomly coloring white areas. low is 10 as a threshold.
    # dimensions are = Red, Blue, Green, Alpha
    # Alpha is set 38 for a ~20% transparency
    mask[..., :][white_areas.T] = (
        np.random.randint(low=10, high=255), np.random.randint(low=10, high=255),
        np.random.randint(low=10, high=255), 38)
    # removing black areas
    mask[..., :][black_areas.T] = 0

    x['mask'] = mask
    return None
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
