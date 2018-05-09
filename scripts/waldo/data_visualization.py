# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0


""" TODO
"""


def visualize_mask(image, masks):
	# image is the backgroun image, and masks is the directory to masks folder
    image = Image.open(image)
    background = image.convert('RGBA')
    mask_background = Image.new('RGBA', (256, 256))
    for filename in masks:
        mask = Image.open(os.getcwd() + '/img/masks/' + filename)
        rgbimg = mask.convert("RGBA")
        data = np.array(rgbimg)  # "data" is a height x width x 4 numpy arrays
        red, green, blue, alpha = data.T  # unpacking blends

        white_areas = (red > 0) & (blue > 0) & (green > 0)
        black_areas = (red == 0) & (blue == 0) & (green == 0) & (alpha == 255)
        # randomly coloring white areas. low is 10 as threshold.
        # dimensions are = Red, Blue, Green, Alpha
        # Alpha is set 127 for a 50% reduced transparency
        data[..., :][white_areas.T] = (
            np.random.randint(low=10, high=255), np.random.randint(low=10, high=255),
            np.random.randint(low=10, high=255), 127)
        # removing black areas
        data[..., :][black_areas.T] = 0

        mask_transformed = Image.fromarray(data, mode='RGBA')
        mask_background = Image.alpha_composite(mask_background, mask_transformed)

    result = Image.alpha_composite(background, mask_background)
    result.show()
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
