#!/usr/bin/env python3

import torch
import argparse
import os
import sys
import random
import numpy as np
from models.Unet import UNet
from train import sample
from dataset import Dataset_madcatar
from waldo.segmenter import ObjectSegmenter
from waldo.core_config import CoreConfig
from waldo.data_visualization import visualize_mask
from unet_config import UnetConfig


parser = argparse.ArgumentParser(description='Pytorch MADCAT Arabic setup')
parser.add_argument('model', type=str,
                    help='path to final model')
parser.add_argument('--dir', default='exp/unet', type=str,
                    help='directory to store segmentation results')
parser.add_argument('--train-dir', default='./data/dev.pth.tar', type=str,
                    help='Path of processed validation data')
parser.add_argument('--train-image-size', default=128, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).'
                    'These are derived from the input images'
                    ' by padding and then random cropping.')
parser.add_argument('--core-config', default='', type=str,
                    help='path of core configuration file')
parser.add_argument('--unet-config', default='', type=str,
                    help='path of network configuration file')
random.seed(0)


def main():
    global args
    args = parser.parse_args()
    args.batch_size = 1  # only segment one image for experiment

    # loading core configuration
    c_config = CoreConfig()
    if args.core_config == '':
        print('No core config file given, using default core configuration')
    if not os.path.exists(args.core_config):
        sys.exit('Cannot find the config file: {}'.format(args.core_config))
    else:
        c_config.read(args.core_config)
        print('Using core configuration from {}'.format(args.core_config))

    # loading Unet configuration
    u_config = UnetConfig()
    if args.unet_config == '':
        print('No unet config file given, using default unet configuration')
    if not os.path.exists(args.unet_config):
        sys.exit('Cannot find the unet configuration file: {}'.format(
            args.unet_config))
    else:
        # need c_config for validation reason
        u_config.read(args.unet_config, args.train_image_size)
        print('Using unet configuration from {}'.format(args.unet_config))

    offset_list = c_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    num_classes = c_config.num_classes
    num_colors = c_config.num_colors
    num_offsets = len(c_config.offsets)
    # model configurations from unet config
    start_filters = u_config.start_filters
    up_mode = u_config.up_mode
    merge_mode = u_config.merge_mode
    depth = u_config.depth

    model = UNet(num_classes, num_offsets,
                 in_channels=num_colors, depth=depth,
                 start_filts=start_filters,
                 up_mode=up_mode,
                 merge_mode=merge_mode)

    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded.")
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    model.eval()  # convert the model into evaluation mode

    val_data = args.train_dir + '/' + 'dev.pth.tar'

    testset = Dataset_madcatar(val_data, c_config, args.train_image_size)
    print('Total samples in the test set: {0}'.format(len(testset)))

    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

    seg_dir = '{}/seg'.format(args.dir)
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    img, class_pred, adj_pred = sample(model, dataloader, seg_dir, c_config)

    seg = ObjectSegmenter(class_pred[0].detach().numpy(),
                          adj_pred[0].detach().numpy()[:2, :, :],
                          num_classes, offset_list[:2], seg_dir)
    mask_pred, object_class = seg.run_segmentation()
    x = {}
    x['img'] = np.moveaxis(img[0].numpy(), 0, -1)
    x['mask'] = mask_pred.astype(int)
    x['object_class'] = object_class
    #visualize_mask(x, c_config)


if __name__ == '__main__':
    main()

