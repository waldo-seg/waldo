#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Yiwen Shao, Desh Raj)
# Apache 2.0

""" This script prepares the training, validation and test data for DSB2018 in a pytorch fashion
"""

import argparse
import random
import torch
from dataset import Dataset_icdar2015

parser = argparse.ArgumentParser(
    description='ICDAR2015 Data Process with Pytorch')
parser.add_argument('--dl_dir', default='/export/b18/draj/icdar_2015/', type=str,
                    help='Path to downloaded dataset')
parser.add_argument('--outdir', default='data', type=str,
                    help='Output directory of processed data')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--train_prop', default=0.9, type=float,
                    help='Propotion of training data after spliting'
                    'the traing input into training and validation data.')


def DataProcess(input_path, train_prop=0.9):
    icdar = Dataset_icdar2015(input_path)
    data = icdar.load_data()

    num_train = len(data['train'])
    train = data['train'][:num_train*train_prop]
    val = data['train'][num_train*train_prop + 1:]
    test = data['test']

    return train, val, test
    

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    train_output = "{0}/train_val/split{1}_seed{2}/train.pth.tar".format(
        args.outdir, args.train_prop, args.seed)
    val_output = "{0}/train_val/split{1}_seed{2}/val.pth.tar".format(
        args.outdir, args.train_prop, args.seed)
    test_output = "{0}/test/test.pth.tar".format(args.outdir)

    random.seed(args.seed)
    train, val, test = DataProcess(args.dl_dir, train_prop=args.train_prop)

    torch.save(train, train_output)
    torch.save(val, val_output)
    torch.save(test, test_output)