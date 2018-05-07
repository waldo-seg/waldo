# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

""" This module provides a pytorch-fashion customized dataset class
"""

import torch
import numpy as np


class Dataset_dsb2018():
    def __init__(self, path, transformation, offset_list,
                 num_classes, height, width):
        # self.data is a dictionary with keys ['img', 'mask'] and probably
        # 'class_object' in the future
        self.data = torch.load(path)
        self.transformation = transformation
        self.offset_list = offset_list
        self.num_classes = num_classes
        self.height = height
        self.width = width

    def __getitem__(self, index):
        data = self.data[index]
        # input images
        img = data['img'].numpy()
        height, width, channel = img.shape
        img = self.transformation(img)

        # bounding box
        num_offsets = len(self.offset_list)
        mask = data['mask'].numpy()
        bound = torch.zeros(num_offsets, self.height, self.width)

        # getting offset feature maps (i.e. bound)
        for k in range(num_offsets):
            i, j = self.offset_list[k]
            # roll the mask in rows and columns according to the offset
            rolled_mask = np.roll(np.roll(mask, i, axis=1), j, axis=0)
            # compare mask and the rolled mask to get whether pixel (x,y) is
            # of the same object as pixel (x+i, y+j)
            bound_unscaled = (torch.FloatTensor(
                (rolled_mask == mask).astype('float'))).unsqueeze(0)
            # do same transformation on the bounds as on images
            bound[k:k + 1] = self.transformation(bound_unscaled)

        # class label
        class_label = torch.zeros((self.num_classes, self.height, self.width))
        for c in range(self.num_classes):
            if c == 0:
                class_label_unscaled = (torch.FloatTensor(
                    (mask == 0).astype('float'))).unsqueeze(0)
            else:  # TODO, the current version is for 2 classes only
                class_label_unscaled = (torch.FloatTensor(
                    (mask > 0).astype('float'))).unsqueeze(0)
            class_label[c:c +
                        1] = self.transformation(class_label_unscaled)

        return img, class_label, bound

    def __len__(self):
        return len(self.data)
