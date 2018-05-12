# Copyright      2018  Johns Hopkins University (author: Daniel Povey, Desh Raj, Adel Rahimi)

# Apache 2.0

import matplotlib.pyplot as plt
import numpy as np
from waldo.data_types import *


def visualize_mask(x, c):
  """This function accepts an object x that should represent an image with a
       mask and a config class c, and it modifies the image to superimpose the "mask" on it.  
       The image will still be visible through a semi-transparent mask layer.
       This function returns None; it modifies x in-place.
    """

    validate_image_with_mask(x, c)
    im = x['img']
    mask = x['mask']
    
    num_objects = np.unique(mask)
    mask_dilated = int(mask*255 / num_objects)
    w, h = mask.shape
    mask_rgb = np.empty((w, h, 3), dtype=np.uint8)
    mask_rgb[:, :, :] = mask_dilated[:, :, np.newaxis]
    
    plt.imshow(im)
    plt.imshow(mask_rgb, alpha=0.2)
    plt.show()

    return

def visualize_polygons(x):
    """This function accepts an object x that should represent an image with
       polygonal objects and it modifies the image to superimpose the edges of
       the polygon on it.
       This function returns None; it modifies x in-place.
    """
    validate_image_with_objects(x)
    # ... do something, modifying x somehow
    return None
