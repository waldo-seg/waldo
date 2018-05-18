# Copyright      2018  Johns Hopkins University (author: Daniel Povey)


# Apache 2.0

from waldo.data_types import *


def randomly_crop_combined_image(combined_image, config,
                                 image_height, image_width):
    """
    This function randomly crops a 'combined image' combined_image
    as would be validated by validate_combined_image(), to an image size
    of 'image_height' by 'image_width'.  It zero-pads if the current
    image is smaller than that.

    It returns the randomly cropped image; it doesn't modify
    'combined_image'.
    """
    validate_combined_image(combined_image)

    cropped_image = 1;  # actualy would be a numpy array.

    validate_combined_image(cropped_image)

    # return cropped_image


def scale_down_image_with_objects(image_with_objects, max_size):
    """
    This function scales down an image with objects (as would be validated by
    validate_image_with_objects(), if the length of its largest side is
    greater than 'max_size'.  (Otherwise it leaves it the same size).

    It modifies the object in-place if necessary; it has no return value.
    """
    pass
