# Copyright      2018  Johns Hopkins University (author: Yiwen Shao)

# Apache 2.0

import os
import numpy as np


class DataSaver:
    def __init__(self, dir):
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            os.makedirs(self.dir + '/numpy_arrays')

    def write_image(self, name, image_with_mask):
        """ This function accepts a image_with_mask object and its name, and saves
            its img, mask and object_class as a numpy array under the given directory (
            i.e. dir/numpy_arrays/name.suffix.npy)
        """
        img = image_with_mask['img']
        mask = image_with_mask['mask']
        obj_class = image_with_mask['object_class']
        path = self.dir + '/numpy_arrays/' + name
        np.save(path + '.img.npy', img)
        np.save(path + '.mask.npy', mask)
        np.save(path + '.object_class.npy', np.array(obj_class))
        self.write_index(name)

    def write_index(self, name):
        """ This function writes image name to a list file named image_ids.txt. It
            contains all the processed image names in order.
        """
        ids_list_name = self.dir + '/' + 'image_ids.txt'
        with open(ids_list_name, 'a') as fh:
            fh.write(name + '\n')
