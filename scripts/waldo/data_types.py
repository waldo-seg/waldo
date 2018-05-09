# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0
import numpy as np


def validate_config(c):
    """This function validates a dict representing certain configuration options for
       our segmentation algorithm and its input.  Note: possibly we should make this
       a class rather than a dict, we can change this later.
       The configuration should satisfy
       the following conditions:
         c['num_classes'] is an integer >= 2, representing the number of object
                     classes.  (This may be a number after augmenting the classes
                     with things like orientation information, so it could be
                     more than the number of 'real' classes of object in the image.).
         c['num_colors'] is an integer >= 1, typically 1 or 3 representing the number of
                     colors in the input images.
         c['offsets'] is an 'offsets list', which is a list of 2-tuples of (x,y)
                  offset co-ordinates.  It should not contain (0,0), and should not
                  contain both (x,y) and (-x,-y).
         c['padding'] is an integer >= 0, representing the amount of zero-padding
                  that we do at the edges of the images when preparing the data.
         c['train_image_size'] is an integer >= 4 * c['padding'].  It represents
                  the width and height of training images (these are assumed to be
                  the same).  Note: training images are manipulated by padding
                  and cropping to this size, not by dilation.

      This function has no return value; on failure, it throws an exception.

      Note: at some point we may decide to make this a class instead of a dict.
    """
    if type(c) != dict:
        raise ValueError('dict type input required.')

    if 'num_classes' not in c or 'num_colors' not in c:
        raise ValueError('num_classes and num_colors required in the dict input.')
    if 'offsets' not in c or 'padding' not in c:
        raise ValueError('offsets and padding required in the dict input.')
    if 'train_image_size' not in c:
        raise ValueError('train_image_size required in the dict input.')

    if type(c['num_classes']) != int or type(c['num_colors']) != int:
        raise ValueError('int type num_classes and num_colors required.')
    if type(c['padding']) != int or type(c['train_image_size']) != int:
        raise ValueError('int type padding and train_image_size required.')

    n_classes, n_colors = c['num_classes'], c['num_colors']
    if n_classes < 2 or n_colors < 1:
        raise ValueError('more than 1 num_classes required.')
    if n_colors < 1:
        raise ValueError('positive num_colors required.')

    val_padding, val_image_size = c['padding'], c['train_image_size']
    if val_padding < 0:
        raise ValueError('non negative padding required.')
    if val_image_size < 4*val_padding:
        raise ValueError('atleast 4 times padding, train_image_size required.')

    offsets_dict = {}
    offsets_list = c['offsets']
    for offset in offsets_list:
        x, y = offset
        if x not in list(offsets_dict.keys()):
            offsets_dict[x] = list()
        offsets_dict[x].append(y)

    for offset in offsets_list:
        x, y = offset
        if x == 0 and y == 0:
            raise ValueError('(0,0) should not be present in offset list')
        if -y in offsets_dict.get(-x, []):
            raise ValueError('(-x,-y) should not be present in offset list')
    return


def validate_image_with_mask(x, c):
    """This function validates an object x that is supposed to represent an image
    with the corresponding object mask.  c is the config object as validated
    by 'validate_config'.  This function returns no value; on failure
    it raises an exception.  Specifically it is checking that:
      x['img'] is a numpy array of shape (num_colors, width, height),
            num_colors is c['num_colors'].
      x['mask'] is an integer numpy array of the same size as x['img'] containing
             integer object-ids from 0 ... num-objects - 1.
      x['object_class'] is a list indexed by object-id giving the class of each object;
            object classes should be in the range 0 .. num_classes - 1, where
            num_classes is c['num_classes'].."""
    validate_config(c)
    if type(x) != dict:
        raise ValueError('dict type input required.')

    if 'img' not in x or 'mask' not in x or 'object_class' not in x:
        raise ValueError('img, mask and object_class required in the dict input.')

    if not isinstance(x['img'], np.ndarray):
        raise ValueError('ndarray type img object required.')
    if not isinstance(x['mask'], np.ndarray):
        raise ValueError('ndarray type mask object required.')
    if not isinstance(x['object_class'], (list,)):
        raise ValueError('list type object_class required.')

    n_classes, n_colors = c['num_classes'], c['num_colors']
    im = x['img']
    dims = im.shape
    if len(dims) != 3:
        raise ValueError('3 dimensional image required.')
    if dims[0] != n_colors:
        raise ValueError('first dimension of np.array should match with config num colors')

    x_mask = x['mask']
    dims_mask = x_mask.shape
    if len(dims_mask) != 2 or dims_mask[0] != dims[1] or dims_mask[1] != dims[2]:
        raise ValueError('same mask shape and image shape required.')

    mask_unique_val = np.unique(x_mask)
    for val in mask_unique_val:
        if type(val) != int:
            raise ValueError('int type mask value required.')

    object_class_list = x['object_class']
    if not set(object_class_list) < set(range(0, n_classes)):
        raise ValueError('object classes between 0 and num_classes required')

    return


def validate_image_with_objects(x, c):
    """This function validates an object x that is supposed to represent an image
    with a list of objects inside it.  c is the config object as validated by
    'validate_config'.  This function has no return value; it raises an
    an exception on failure.

     Specifically it is checking that:
      x['img'] is a numpy array of shape (num_colors, width, height),
           where num_colors is c['num_colors'].
      x['objects'] is a sorted list of elements y satisfying validate_object(y).
    """
    validate_config(c)
    if type(x) != dict:
        raise ValueError('dict type input required.')

    if 'img' not in x or 'objects' not in x:
        raise ValueError('img and objects required in the dict input.')

    if not isinstance(x['img'], np.ndarray):
        raise ValueError('img of numpy array type required.')

    if not isinstance(x['objects'], (list,)):
        raise ValueError('objects of list type required.')

    n_colors = c['num_colors']
    im = x['img']
    dims = im.shape
    if len(dims) != 3:
        raise ValueError('3 dimensional image required.')
    if dims[0] != n_colors:
        raise ValueError('first dimension of np.array should match with config num colors')

    return


def validate_object(x):
    """This function validates an object x that is supposed to represent an object
    inside an image, and throws an exception on failure.
    Specifically it is checking that:
      x['polygon'] is a list of >= 3 integer (x,y) pairs representing the corners
                    of the polygon in clockwise or anticlockwise order.
    (We may add further checks later on).
    """
    if type(x) != dict:
        raise ValueError('dict type input required.')

    if 'polygon' not in x:
        raise ValueError('polygon object required.')

    if not isinstance(x['polygon'], (list,)):
        raise ValueError('list type polygon object required.')

    points_list = x['polygon']
    if len(points_list) < 3:
        raise ValueError('More than two points required.')

    for x, y in points_list:
        if type(x) != int or type(y) != int:
            raise ValueError('integer (x,y) pairs required.')

    return


def validate_polygon(x):
    """This function validates an object x that is supposed to represent a polygon.
    This should be a list of >= 3 integer (x,y) pairs representing the corners
    of a polygon in clockwise or anticlockwise order.
    This function has no return value; it raises an exception on failure."""

    points_list = x
    if len(points_list) < 3:
        raise ValueError('More than two points required.')

    for x, y in points_list:
        if type(x) != int or type(y) != int:
            raise ValueError('integer (x,y) pairs required.')
    return


def validate_combined_image(x, c):
    """This function validates a 'combined image'.  A 'combined image' is a numpy
    array that contains both input and output information, ready for further
    preprocessing and eventually neural network training (although we'll split it
    up before we actually train the network.

    A combined image should be a numpy array with shape (dim, width, height),
    where 'dim' equals num_colors + 2 * (num_classes + num_offsets)
    where num_colors, num_classes and num_offsets are derived from the
    configuration object 'c'.

    The meaning of the combined image is as follows:
      x[0:num_colors,...] is the input image
    Let 'num_outputs' equal num_classes + num_offsets.
      x[num_colors:num_colors+num_outputs,...] is the 'positive labels' for
          each class or offset, which will be 1 if it's that class or if
          it's the same object at that offset, and 0 otherwise.
      x[num_colors+num_outputs:,...] is the 'negative labels' for
          each class or offset, which will be 1 if it is known not to be
          that class (or if the object at that offset is different), and
          zero otherwise.  In the zero-padded part of the image or close to the
          edges of training images, both positive and negative labels will
          be zero.
    """
    validate_config(c)

    if not isinstance(x, np.ndarray):
        raise ValueError('x of numpy array type required.')

    dims = x.shape
    if len(dims) != 3:
        raise ValueError('3 dimensional image required.')

    n_colors = c['num_colors']
    n_classes = c['num_classes']
    n_offsets = c['num_offsets']

    dim = n_colors + 2 * (n_classes + n_offsets)
    if dims[0] != dim:
        raise ValueError('first dimension of np.array should match with num_colors + 2 * (num_classes + num_offsets)')

    n_outputs = n_classes + n_offsets
    x1 = x[n_colors:n_colors+n_outputs, :, :]
    unique_val = np.unique(x1)
    if not set(unique_val) <= set(range(0, 2)):
        raise ValueError('unique values 0, 1 expected)')

    x1 = x[n_colors + n_outputs:, :, :]
    unique_val = np.unique(x1)
    if not set(unique_val) <= set(range(0, 2)):
        raise ValueError('unique values 0, 1 expected)')

    return
