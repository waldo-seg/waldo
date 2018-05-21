#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

""" This module will be used for creating text localization mask on page image.
 Given the word segmentation (bounding box around a word) for every word, it will
 extract line segmentation. To extract line segmentation, it will take word bounding
 boxes of a line as input, will create a minimum area bounding box that will contain
 all corner points of word bounding boxes. The obtained bounding box (will not necessarily
 be vertically or horizontally aligned).
"""

import torch
import xml.dom.minidom as minidom
from torch.utils.data import Dataset, DataLoader
from waldo.mar_utils import compute_hull
from waldo.data_manipulation import convert_to_combined_image
from waldo.data_transformation import randomly_crop_combined_image
from waldo.core_config import CoreConfig


class Dataset_madcatar(Dataset):

    def __init__(self, path, c_cfg, size):
        # self.data is a dictionary with keys ['id', 'img', 'mask', 'object_class']
        self.data = torch.load(path)
        self.c_cfg = c_cfg
        self.size = size

    def __getitem__(self, index):
        data = self.data[index]
        combined_img = convert_to_combined_image(data, self.c_cfg)
        n_classes = self.c_cfg.num_classes
        n_offsets = len(self.c_cfg.offsets)
        n_colors = self.c_cfg.num_colors
        cropped_img = randomly_crop_combined_image(
            combined_img, self.c_cfg, self.size, self.size)

        img = torch.from_numpy(
            cropped_img[:n_colors, :, :]).type(torch.FloatTensor)
        class_label = torch.from_numpy(
            cropped_img[n_colors:n_colors + n_classes, :, :]).type(torch.FloatTensor)
        bound = torch.from_numpy(
            cropped_img[n_colors + n_classes:n_colors +
                        n_classes + n_offsets, :, :]).type(torch.FloatTensor)

        return img, class_label, bound

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import torchvision
    c_config = CoreConfig()
    c_config.read('exp/unet_5_10_sgd/configs/core.config')
    trainset = Dataset_madcatar('data/train.pth.tar',
                               c_config, 128)
    trainloader = DataLoader(
        trainset, num_workers=1, batch_size=16, shuffle=True)
    data_iter = iter(trainloader)
    # data_iter.next()
    img, class_label, bound = data_iter.next()
