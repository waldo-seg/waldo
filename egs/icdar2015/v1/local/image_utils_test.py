#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Desh Raj)
# Apache 2.0

""" 
Test cases for visualizing image with mask and compressing image with mask. 
"""

from dataset import DatasetICDAR2015
from waldo.data_visualization import visualize_mask
from waldo.data_manipulation import compress_image_with_mask
from waldo.core_config import CoreConfig

import sys
import random
import numpy as np
import unittest

# DL_DIR = '/export/b18/draj/icdar_2015/'
DL_DIR = '/home/desh/Research/icdar/icdar_2015/sample/'
TRANSPARENCY = 0.3

class ImageUtilsTest(unittest.TestCase):
    """Testing image utilities: visualization and compression
    """
    def setUp(self):
        """This method sets up objects for all the test cases.
        """
        icdar = DatasetICDAR2015(DL_DIR)
        data = icdar.load_data()
        self.test_object = random.choice(data['test'])

        self.c = CoreConfig()
        self.c.num_colors = self.test_object['img'].shape[2]

        self.transparency = TRANSPARENCY


    def test_visualize_object(self):
        """Given a dictionary object as follows
        x['img']: numpy array of shape (height,width,colors)
        x['mask']: numpy array of shape (height,width), with every element categorizing it 
        into one of the object ids
        The method generates an image overlaying a translucent mask on the image.
        """
        visualize_mask(self.test_object, self.c, self.transparency)


    def test_compress_object(self):
        """Given a dictionary object as follows
        x['img']: numpy array of shape (height,width,colors)
        x['mask']: numpy array of shape (height,width), with every element categorizing it 
        into one of the object ids
        The method compresses the object and prints the original and compressed sizes.
        It also asserts that the original size should be greater than the compressed size.
        """
        y = compress_image_with_mask(self.test_object,self.c)
        x_mem = sys.getsizeof(self.test_object)
        y_mem = sys.getsizeof(y)
        self.assertTrue(y_mem <= x_mem)


if __name__ == '__main__':
    unittest.main()
    