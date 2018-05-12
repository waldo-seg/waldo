# Copyright      2018  Johns Hopkins University (author: Daniel Povey, Desh Raj, Adel Rahimi)

# Apache 2.0

import matplotlib.pyplot as plt
import numpy as np
from waldo.data_types import *


def visualize_mask(x, c, transparency):
    """
    This function accepts an object x that should represent an image with a
    mask, a config class c, and a float 0 < transparency < 1.  
    It displays the mask overlay on the image with the mask transparency 
    described by the parameter.
    """

    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.
        '''
        return plt.cm.get_cmap(name, n)


    def get_colored_mask(mask, n, cmap):
        """Given a BW mask, number of objects, and a LinearSegmentedColormap object, 
        returns a RGB mask.
        """
        color_mask = np.array([cmap(i) for i in mask])
        return np.array(color_mask)



    validate_image_with_mask(x, c)
    im = x['img']
    im = np.swapaxes(im, 0, 2)
    mask = np.transpose(x['mask'])
    
    num_objects = np.unique(mask).shape[0]
    cmap = get_cmap(num_objects)
    mask_rgb = get_colored_mask(mask,num_objects,cmap)

    plt.imshow(im)
    plt.imshow(mask_rgb, alpha=transparency)
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
