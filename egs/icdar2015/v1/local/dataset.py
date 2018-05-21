#!/usr/bin/env python3

# Copyright   2018 Johns Hopkins University (author: Desh Raj)
# Apache 2.0

""" This module will be used for loading ICDAR 2015 data as a class.
 Further functions may be added as and when required.
"""

import os
import numpy as np
from PIL import Image,ImageDraw
from glob import glob
from waldo.data_manipulation import * 
from waldo.core_config import CoreConfig


class DatasetICDAR2015:
    """Class to load and process the ICDAR 2015 data.
    Initialize with path to directory containing downloaded dataset.
    The load_data() function can be used after initialization. It returns
    a dict containing two lists: train and test.

    Elements in both lists are dicts x as follows:
    x['img'] is a numpy array of shape (num_colors, width, height),
            num_colors is c.num_colors.
    x['mask'] is an integer numpy array of the same size as x['img'] containing
             integer object-ids from 0 ... num-objects - 1.
    """

    TRAIN_IMAGES = "train/images"
    TEST_IMAGES = "test/images"
    TRAIN_LABELS = "train/labels"
    TEST_LABELS = "test/labels"

    

    def __init__(self, data_dir):
        """Constructor for the class.
        Validates the path to ICDAR 2015 data.
        """
        self.data_dir=""
        
        self.core_config = CoreConfig()
        self.core_config.num_colors = 3

        if not self._validate_path(data_dir):
            raise ValueError("The path is invalid. Either of the following"
                " could be wrong:\n"
                "- Path does not exist.\n"
                "- Path does not point to a directory.\n"
                "- Directory is empty.\n"
                "- The training or test directories have unequal number"
                "of images and labels leading to a mismatch.")
        else:
            self.data_dir = data_dir


    def load_data(self):
        """Loads the ICDAR data and returns a dict containing two structures: train and test
        Returns
        --------
        data (image,image): train and test image lists
        """
        train_data = self._load_data_worker(self.tr_img_dir, self.tr_lbl_dir)
        test_data = self._load_data_worker(self.te_img_dir, self.te_lbl_dir)

        data = {
            'train':train_data,
            'test':test_data
        }

        return data


    def _check_images_and_labels(self, image_dir, label_dir):
        """Checks if the number of images is equal to
        the number of labels in the path.
        Input
        ------
        image_dir (string): path to image directory
        label_dir (string): path to label directory
        """
        return len(os.listdir(image_dir))==len(os.listdir(label_dir))


    def _validate_path(self, data_dir):
        """Checks path for validity.
        Returns
        --------
        is_valid (boolean): True, if path is valid
        """
        if (os.path.exists(data_dir) 
            and os.path.isdir(data_dir)
            and os.listdir(data_dir)):

            self.tr_img_dir = data_dir + self.TRAIN_IMAGES
            self.tr_lbl_dir = data_dir + self.TRAIN_LABELS
            self.te_img_dir = data_dir + self.TEST_IMAGES
            self.te_lbl_dir = data_dir + self.TEST_LABELS

            if (self._check_images_and_labels(self.tr_img_dir, self.tr_lbl_dir) 
                and self._check_images_and_labels(self.te_img_dir, self.te_lbl_dir)):
                
                return True
        
        return False



    def _load_data_worker(self,img_dir,lbl_dir):
        """Given the image and label directories, returns a structured list.
        Input
        ------
        img_dir (string): path to image directory
        lbl_dir (string): path to label directory

        Returns
        -------
        data (list(image)): list containing image data
        """
        data = []

        for img,lbl in zip(glob(img_dir+"/*.jpg"),glob(lbl_dir+"/*.txt")):
            im = Image.open(img)
            im_arr = np.array(im)
            lbl_fh = open(lbl,encoding='utf-8')

            objects = self._get_objects(lbl_fh)
            sorted_objects = sort_object_list(objects)
            object_class = self._get_object_classes(sorted_objects)
            
            image_with_objects = {
                'img':im_arr,
                'objects':sorted_objects,
                'object_class': object_class
            }

            image_with_mask = convert_to_mask(image_with_objects, self.core_config)

            data.append(image_with_mask)
            lbl_fh.close()

        return data


    def _get_objects(self,label_fh):
        """ Given the file handle of the file containing image data, it returns
        a list of the objects contained in the image.
        Returns
        -------
        objects (list(object)), where object is a dict. Currently, object has only
        one key, namely 'polygon' which is a list of points in clockwise order.
        """
        objects = []
        for line in label_fh.readlines():
            try:
                object = {}
                line = line.replace(u'\ufeff', '')
                if line != '':
                    x1, y1, x2, y2, x3, y3, x4, y4= [int(i) for i in line.split(',')[:-1]]
                    p1 = (x1, y1)
                    p2 = (x2, y2)
                    p3 = (x3, y3)
                    p4 = (x4, y4)
                    object['polygon'] = [p1,p2,p3,p4]
                    objects.append(object)
            except:
                pass
        return objects


    def _get_object_classes(self,objects):
        """Given the list of objects, it returns an array mapping object ids to their
        respective classes. Background has class 0 and text has class 1.
        """
        object_class = [1 for object in objects]
        object_class.insert(0, 0)
        return object_class
