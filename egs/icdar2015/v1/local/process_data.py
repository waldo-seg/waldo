#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Yiwen Shao, Desh Raj)
# Apache 2.0

""" This script prepares the training, validation and test data for ICDAR2015 in a pytorch fashion
"""

import os
import argparse
import random
import torch
from dataset import DatasetICDAR2015
from waldo.data_io import DataSaver
from waldo.core_config import CoreConfig


parser = argparse.ArgumentParser(
    description='ICDAR2015 Data Process with Pytorch')
parser.add_argument('--dl_dir', default='/export/b18/draj/icdar_2015', type=str,
                    help='Path to downloaded dataset')
parser.add_argument('--outdir', default='data', type=str,
                    help='Output directory of processed data')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--train_prop', default=0.9, type=float,
                    help='Propotion of training data after spliting'
                    'the traing input into training and validation data.')
parser.add_argument('--cfg', default='data/core.config', type=str,
                    help='core config file for preparing data')


def DataProcess(input_path, cfg, train_prop=0.9):
    icdar = DatasetICDAR2015(input_path, cfg)
    print ('Loading ICDAR data from disk...')
    train_test = icdar.load_data()
    train_test_ids = icdar.get_image_ids()

    train_data = list(zip(train_test['train'],train_test_ids['train']))
    random.shuffle(train_data)
    train_test['train'][:], train_test_ids['train'][:] = zip(*train_data)

    print ('Preparing training and validation splits in ratio {}...'.format(train_prop))
    num_total = len(train_test['train'])
    num_train = int(num_total*train_prop)
    data = {
        'train': train_test['train'][:num_train],
        'val': train_test['train'][num_train+1:],
        'test': train_test['test']
    }
    data_ids = {
        'train': train_test_ids['train'][:num_train],
        'val': train_test_ids['train'][num_train+1:],
        'test': train_test_ids['test']
    }

    return data, data_ids


def save_data(data, data_ids, outdir, split):
    print ('Saving {} data...'.format(split))
    saver = DataSaver(os.path.join(outdir, split), cfg, train=(split!='test'))
    for item,id in zip(data,data_ids):
        saver.write_image(id, item)
    saver.write_index()
    

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    cfg = CoreConfig()
    cfg.read(args.cfg)

    data, data_ids = DataProcess(args.dl_dir, cfg, train_prop=args.train_prop)

    for split in ['train','val','test']:
        save_data(data[split], data_ids[split], args.outdir, split)

    print ('Finished processing data.')
    
