#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Desh Raj)
# Apache 2.0

""" 
Example script to visualize the mask generated by the data loader class.
Picks up a random image object from the training set and displays the mask overlay. 
"""

from dataset import DatasetICDAR2015
from waldo.data_visualization import visualize_mask
from waldo.core_config import CoreConfig

import argparse
import random
import numpy as np


def visualize_object(x, transparency):
    """Given a dictionary object as follows
    x['img']: numpy array of shape (num_class,width,height)
    x['mask']: numpy array of same dimensions as image, but with every element categorizing it 
    into one of the object ids
    The method generates an image overlaying a translucent mask on the image and displays it.
    """
    c = CoreConfig()
    c.num_colors = x['img'].shape[0]
    visualize_mask(x,c,transparency)
    return


parser = argparse.ArgumentParser(description='ICDAR2015 image mask visualization')
parser.add_argument('--dl_dir', default='/export/b18/draj/icdar_2015/', type=str,
                    help='Path to downloaded dataset')
parser.add_argument('--transparency', default=0.3, type=float,
                    help='Transparency of mask. Takes values between 0 and 1.')


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    transparency = args.transparency
    if (transparency > 1 or transparency < 0):
        transparency = 0

    input_path = args.dl_dir

    icdar = DatasetICDAR2015(input_path)
    data = icdar.load_data()

    # x = random.choice(data['train'])
    x = data['test'][1]
    visualize_object(x,transparency)

