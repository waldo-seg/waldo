# Copyright      2018  Johns Hopkins University (author: Yiwen Shao)

# Apache 2.0

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from waldo.data_manipulation import convert_to_combined_image
from waldo.data_transformation import randomly_crop_combined_image


class DataSaver:
    def __init__(self, dir):
        self.dir = dir
        self.suffixes = ['img', 'mask', 'object_class']
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            for suffix in self.suffixes:
                subdir = os.path.join(self.dir, suffix)
                os.makedirs(subdir)
        self.ids = []

    def write_image(self, name, image_with_mask):
        """ This function accepts a image_with_mask object and its name, and saves
            its img, mask and object_class as a numpy array under the given directory (
            i.e. dir/numpy_arrays/name.suffix.npy)
        """
        self.__validate_name(name)

        for suffix in self.suffixes:
            path = os.path.join(self.dir, suffix)
            filename = path + '/' + name + '.' + suffix + '.npy'
            if suffix == 'object_class':
                # object_class is a list not a numpy array
                np.save(filename, np.array(image_with_mask[suffix]))
            else:
                np.save(filename, image_with_mask[suffix])

        self.ids.append(name)

    def write_index(self):
        """ This function writes image name to a list file named image_ids.txt. It
            contains all the processed image names in order.
        """
        ids_filename = self.dir + '/' + 'image_ids.txt'
        with open(ids_filename, 'w') as fh:
            for id in self.ids:
                fh.write(id + '\n')

    def __validate_name(self, name):
        if name.find(' ') != -1 or name.find('/') != -1:
            raise ValueError(
                'image id should not contain subspace or slash but get {}'.format(name))
        if name in self.ids:
            raise ValueError('get duplicated image id: {}'.format(name))
        return


class WaldoDataset(Dataset):
    def __init__(self, dir, c_cfg, size, cache=True):
        self.c_cfg = c_cfg
        self.size = size
        self.dir = dir
        self.cache = cache
        self.data = []
        with open(self.dir + '/' + 'image_ids.txt', 'r') as ids_file:
            self.ids = ids_file.readlines()
        self.ids = [id.strip() for id in self.ids]
        # cache everything into memory if True
        if self.cache:
            for id in self.ids:
                image_with_mask = self.__load_data(id)
                self.data.append(image_with_mask)

    def __load_data(self, id):
        suffixes = ['img', 'mask', 'object_class']
        image_with_mask = {}
        for suffix in suffixes:
            path = os.path.join(self.dir, suffix)
            filename = path + '/' + id + '.' + suffix + '.npy'
            if suffix == 'object_class':
                image_with_mask[suffix] = np.load(filename).tolist()
            else:
                image_with_mask[suffix] = np.load(filename)
        return image_with_mask

    def __getitem__(self, index):
        if self.cache:
            image_with_mask = self.data[index]
        else:
            id = self.ids[index]
            image_with_mask = self.__load_data(id)
        combined_img = convert_to_combined_image(image_with_mask, self.c_cfg)
        n_classes = self.c_cfg.num_classes
        n_offsets = len(self.c_cfg.offsets)
        n_colors = self.c_cfg.num_colors
        cropped_img = randomly_crop_combined_image(
            combined_img, self.c_cfg, self.size, self.size)

        img = torch.from_numpy(
            cropped_img[:n_colors, :, :])
        class_label = torch.from_numpy(
            cropped_img[n_colors:n_colors + n_classes, :, :])
        bound = torch.from_numpy(
            cropped_img[n_colors + n_classes:n_colors +
                        n_classes + n_offsets, :, :])

        return img, class_label, bound

    def __len__(self):
        return len(self.ids)
