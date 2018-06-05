# Copyright      2018  Johns Hopkins University (author: Daniel Povey, Desh Raj, Adel Rahimi)
#                      Hossein Hadian

# Apache 2.0
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import operator

from waldo.data_types import *


def visualize_mask(x, c, transparency=0.7, show_labels=True):
    """
    This function accepts an object x that should represent an image with a
    mask, a config class c, and a float 0 < transparency < 1.
    It changes the image in-place by overlaying the mask with transparency
    described by the parameter.
    x['img_with_mask'] = image with transparent mask overlay
    """

    validate_image_with_mask(x, c)
    im = x['img']
    mask = x['mask']

    num_objects = np.unique(mask).shape[0]
    colormap = plt.cm.get_cmap('hsv', num_objects)

    shuffled_object_ids = np.random.permutation(np.array(list(range(num_objects))))
    color_mask = np.array([colormap(shuffled_object_ids[i]) for i in mask])
    obj2center = {}
    obj2size = {}
    for row in range(len(mask)):
        for col in range(len(mask[row])):
            objid = mask[row, col]
            obj2center[objid] = (obj2center[objid] + (row, col) if objid in
                                 obj2center else np.array((row, col)))
            obj2size[objid] = (obj2size[objid] + 1 if objid in obj2size
                               else 1)
            objclass = x['object_class'][objid]
            # set any background to white:
            if objclass == 0: color_mask[row, col] = [1, 1, 1, 1]
    for objid in obj2center:
        obj2center[objid] = tuple(np.round(obj2center[objid] /
                                           obj2size[objid]).astype(int))
    plt.clf()
    plt.imshow(im)
    plt.imshow(color_mask, alpha=transparency)
    if show_labels:
        for objid in obj2center:
            center = obj2center[objid]
            color = colormap(shuffled_object_ids[objid])
            plt.text(center[1] - 2, center[0] + 2, '{}'.format(objid), fontsize=7,
                     color=color, bbox=dict(facecolor='white',
                                            edgecolor='none', pad=0))

    plt.subplots_adjust(0,0,1,1)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = Image.open(buffer_)
    x['img_with_mask'] = np.array(image)
    buffer_.close()
    return x



def visualize_polygons(x):
    """This function accepts an object x that should represent an image with
       polygonal objects and it modifies the image to superimpose the edges of
       the polygon on it.
       This function returns None; it modifies x in-place.
    """
    validate_image_with_objects(x)
    # ... do something, modifying x somehow
    return None
