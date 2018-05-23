#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

""" This script prepares the training, validation and test data for DSB2018 in a pytorch fashion
"""

import os
import sys
import argparse
import random
import torch
import numpy as np
from PIL import Image
from waldo.data_io import DataSaver

parser = argparse.ArgumentParser(
    description='DSB2018 Data Process with Pytorch')
parser.add_argument('--train-input', default='data/download/stage1_train/', type=str,
                    help='Path of raw training data')
parser.add_argument('--test-input', default='data/download/stage1_test/', type=str,
                    help='Path of raw test data')
parser.add_argument('--outdir', default='data', type=str,
                    help='Output directory of processed data')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--train-prop', default=0.9, type=float,
                    help='Propotion of training data after spliting'
                    'the traing input into training and validation data.')
parser.add_argument('--img-channels', default=3, type=int,
                    help='Number of channels for input images')


def DataProcess(input_path, output_dir, channels, mode='train', train_prop=0.9):
    if mode == 'train':
        # Get train IDs
        train_all_ids = next(os.walk(input_path))[1]
        # split the training set into train and validation set
        random.shuffle(train_all_ids)

        num_all = len(train_all_ids)
        num_train = int(num_all * train_prop)
        train_ids = train_all_ids[:num_train]
        val_ids = train_all_ids[num_train:]

        train_saver = DataSaver(output_dir + '/' + 'train')

        print('Getting train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in enumerate(train_ids):
            train_item = {}
            train_item['name'] = id_
            path = input_path + '/' + id_
            img = np.array(Image.open(path + '/images/' + id_ +
                                      '.png'))[:, :, :channels]

            train_item['img'] = img
            mask = None
            object_id = 1
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = (np.array(Image.open(path + '/masks/' + mask_file)
                                  ) / 255).astype('uint8')
                if mask is None:
                    mask = mask_
                else:
                    mask = np.maximum(mask, mask_ * object_id)
                object_id += 1
            train_item['mask'] = mask
            # only object ID 0 belongs to background
            object_class = np.ones(object_id)
            object_class[0] = 0
            train_item['object_class'] = object_class.tolist()
            train_saver.write_image(train_item['name'], train_item)

        val_saver = DataSaver(output_dir + '/' + 'val')
        print('Getting validation images and masks ... ')
        sys.stdout.flush()
        for n, id_ in enumerate(val_ids):
            val_item = {}
            val_item['name'] = id_
            path = input_path + '/' + id_
            img = np.array(Image.open(path + '/images/' +
                                      id_ + '.png'))[:, :, :channels]

            val_item['img'] = img
            mask = None
            object_id = 1
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = (np.array(Image.open(path + '/masks/' + mask_file)
                                  ) / 255).astype('uint8')
                if mask is None:
                    mask = mask_
                else:
                    mask = np.maximum(mask, mask_ * object_id)
                object_id += 1
            val_item['mask'] = mask
            object_class = np.ones(object_id)
            object_class[0] = 0
            val_item['object_class'] = object_class.tolist()
            val_saver.write_image(val_item['name'], val_item)

        print('Done with training and validation set!')

    else:
        # Get test images
        print('Getting test images ... ')
        test_ids = next(os.walk(input_path))[1]
        test = []
        sys.stdout.flush()
        for n, id_ in enumerate(test_ids):
            test_item = {}
            test_item['name'] = id_
            path = input_path + '/' + id_
            img = np.array(Image.open(path + '/images/' +
                                      id_ + '.png'))[:, :, :channels]
            test_item['img'] = img
            test.append(test_item)

        print('Done with test set!')


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    train_val_dir = args.outdir + '/train_val'
    train_ids_file = "{0}/train/image_ids.txt".format(train_val_dir)
    val_ids_file = "{0}/val/image_ids.txt".format(train_val_dir)
    if not (os.path.exists(train_ids_file) and
            os.path.exists(val_ids_file)):
        random.seed(args.seed)
        DataProcess(args.train_input,
                    train_val_dir,
                    args.img_channels, mode='train',
                    train_prop=args.train_prop)

    else:
        print('Not processing training and validation data as it is already there.')

    # test_output = "{0}/test/test.pth.tar".format(args.outdir)
    # if not (os.path.exists(test_output)):
    #     test = DataProcess(args.test_input, args.img_channels, mode='test')
    #     torch.save(test, test_output)
    # else:
    #     print('Not processing test data as it is already there.')
