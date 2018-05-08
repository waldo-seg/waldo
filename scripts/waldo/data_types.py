# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0


""" TODO
"""

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
    return


def validate_image_with_mask(x, c)
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
    # TODO.
    return


def validate_image_with_objects(x, c):
    """This function validates an object x that is supposed to represent an image
    with a list of objects inside it.  c is the config object as validated by
    'validate_config'.  This function has no return value; it raises an
    an exception on failure.

     Specifically it is checking that:
      x['img'] is a numpy array of shape (num_colors, width, height),
           where num_colors is c['num_colors'].
      x['objects'] is a list of elements y satisfying validate_object(y).
    """
    validate_config(c)
    # TODO.
    return


def validate_object(x):
    """This function validates an object x that is supposed to represent an object
    inside an image, and throws an exception on failure.
    Specifically it is checking that:
      x['polygon'] is a list of >= 3 integer (x,y) pairs representing the corners
                    of the polygon in clockwise or anticlockwise order.
    (We may add further checks later on).
    """
    # TODO.
    return


def validate_polygon(x):
    """This function validates an object x that is supposed to represent a polygon.
    This should be a list of >= 3 integer (x,y) pairs representing the corners
    of a polygon in clockwise or anticlockwise order.
    This function has no return value; it raises an exception on failure."""
    # TODO.
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
    # TODO.
    return None
