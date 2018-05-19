# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

""" This module provides a pytorch-fashion customized dataset class
"""

import torch
from torch.utils.data import Dataset, DataLoader
from waldo.data_manipulation import convert_to_combined_image
from waldo.data_transformation import randomly_crop_combined_image


class Dataset_dsb2018(Dataset):
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
    from waldo.core_config import CoreConfig
    import torchvision
    c_config = CoreConfig()
    c_config.read('exp/unet_5_10_sgd/configs/core.config')
    trainset = Dataset_dsb2018('data/train_val/train.pth.tar',
                               c_config, 128)
    trainloader = DataLoader(
        trainset, num_workers=1, batch_size=16, shuffle=True)
    data_iter = iter(trainloader)
    # data_iter.next()
    img, class_label, bound = data_iter.next()
    # torchvision.utils.save_image(class_label[:, 0:1, :, :], 'class0.png')
    # torchvision.utils.save_image(class_label[:, 1:2, :, :], 'class1.png')
    # torchvision.utils.save_image(bound[:, 0:1, :, :], 'bound0.png')
    # torchvision.utils.save_image(bound[:, 1:2, :, :], 'bound1.png')
