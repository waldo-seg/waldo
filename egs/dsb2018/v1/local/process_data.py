#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

""" This script prepares the training, validation and test data for DSB2018 in a pytorch fashion
"""

import os
import sys
import argparse
import random
import numpy as np
from PIL import Image
from waldo.data_io import DataSaver
from waldo.data_transformation import make_square_image_with_padding
from waldo.core_config import CoreConfig

parser = argparse.ArgumentParser(
    description='DSB2018 Data Process with Pytorch')
parser.add_argument('--indir', default='data/download', type=str,
                    help='Path to extracted raw data')
parser.add_argument('--outdir', default='data', type=str,
                    help='Output directory of processed data')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--train-prop', default=0.9, type=float,
                    help='Propotion of training data after spliting'
                    'the traing input into training and validation data.')
parser.add_argument('--cfg', default='data/core.config', type=str,
                    help='core config file for preparing data')


def DataProcess(input_dir, output_dir, split_name, cfg, train_prop=0.9):
    channels = cfg.num_colors
    split_dir = os.path.join(input_dir, split_name)
    if split_name == 'train':
        split_dir = os.path.join(input_dir, 'stage1_train')
        # Get train IDs
        train_all_ids = next(os.walk(split_dir))[1]
        # split the training set into train and validation set
        random.shuffle(train_all_ids)

        num_all = len(train_all_ids)
        num_train = int(num_all * train_prop)
        train_ids = train_all_ids[:num_train]
        val_ids = train_all_ids[num_train:]

        train_saver = DataSaver(os.path.join(output_dir, 'train'), cfg)

        print('Getting train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in enumerate(train_ids):
            train_item = {}
            path = split_dir + '/' + id_
            img = np.array(Image.open(path + '/images/' + id_ +
                                      '.png'))[:, :, :channels]
            # padding if not square
            img = make_square_image_with_padding(img, channels)
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
            mask = make_square_image_with_padding(mask, 1)
            train_item['mask'] = mask
            # only object ID 0 belongs to background
            object_class = np.ones(object_id)
            object_class[0] = 0
            train_item['object_class'] = object_class.tolist()
            train_saver.write_image(id_, train_item)
        # write all ids to a file named 'image_ids.txt'
        train_saver.write_index()

        val_saver = DataSaver(os.path.join(output_dir, 'val'), cfg)
        print('Getting validation images and masks ... ')
        sys.stdout.flush()
        for n, id_ in enumerate(val_ids):
            val_item = {}
            path = split_dir + '/' + id_
            img = np.array(Image.open(path + '/images/' +
                                      id_ + '.png'))[:, :, :channels]

            # padding if not square
            img = make_square_image_with_padding(img, channels)
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
            mask = make_square_image_with_padding(mask, 1)
            val_item['mask'] = mask
            object_class = np.ones(object_id)
            object_class[0] = 0
            val_item['object_class'] = object_class.tolist()
            val_saver.write_image(id_, val_item)

        val_saver.write_index()
        print('Done with training and validation set!')

    else:
        # Get test images
        test_saver = DataSaver(os.path.join(
            output_dir, split_name), cfg, train=False)
        print('Getting {} images ... '.format(split_name))
        test_ids = next(os.walk(split_dir))[1]
        sys.stdout.flush()
        for n, id_ in enumerate(test_ids):
            test_item = {}
            path = split_dir + '/' + id_
            img = np.array(Image.open(path + '/images/' +
                                      id_ + '.png'))
            if len(img.shape) == 2 and channels == 3:
                # expand and reshape it to size:(height, width, channels)
                img = np.moveaxis(np.array([img, img, img]), 0, -1)
            else:
                img = img[:, :, :channels]
            test_item['img'] = img
            test_saver.write_image(id_, test_item)
        test_saver.write_index()
        print('Done with {} set!'.format(split_name))


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    cfg = CoreConfig()
    cfg.read(args.cfg)

    split_names = ['train', 'stage1_test', 'stage2_test_final']
    for split in split_names:
        ids_file = "{0}/{1}/image_ids.txt".format(args.outdir, split)
        if not (os.path.exists(ids_file)):
            random.seed(args.seed)
            DataProcess(args.indir, args.outdir, split,
                        cfg, train_prop=args.train_prop)
        else:
            print('Not processing {} data as it is already there.'.format(split))
