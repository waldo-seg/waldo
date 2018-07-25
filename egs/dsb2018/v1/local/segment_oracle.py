#!/usr/bin/env python3

import torch
import csv
import argparse
import os
import random
import numpy as np
import scipy.misc
import sys
import time

sys.path.insert(0, 'scripts')
from models.Unet import UNet
from waldo.segmenter import ObjectSegmenter, SegmenterOptions
from skimage.transform import resize
from waldo.core_config import CoreConfig
from waldo.data_visualization import visualize_mask
from waldo.data_io import WaldoDataset
from unet_config import UnetConfig
import csegment.c_segment as cseg


parser = argparse.ArgumentParser(description='Pytorch DSB2018 setup')
parser.add_argument('--test-data', type=str, required=True,
                    help='Path to test images to be segmented')
parser.add_argument('--dir', type=str, required=True,
                    help='Directory to store segmentation results. '
                    'It is assumed that <dir> is a sub-directory of '
                    'the model directory.')
parser.add_argument('--train-image-size', default=128, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).'
                    'These are derived from the input images'
                    ' by padding and then random cropping.')
parser.add_argument('--object-merge-factor', type=float, default=None,
                    help='Scale for object merge scores in the segmentaion '
                    'algorithm. If not set, it will be set to '
                    '1.0 / num_offsets by default.')
parser.add_argument('--same-different-bias', type=float, default=0.0,
                    help='Bias for same/different probs in the segmentation '
                    'algorithm.')
parser.add_argument('--merge-logprob-bias', type=float, default=0.0,
                    help='A bias that is added to merge logprobs in the '
                    'segmentation algorithm.')
parser.add_argument('--prune-threshold', type=float, default=0.0,
                    help='Threshold used in the pruning step of the '
                    'segmentation algorithm. Higher values --> more pruning.')
parser.add_argument('--csv', type=str, default='sub-dsbowl2018.csv',
                    help='Csv filename as the final submission file')
parser.add_argument('--job', type=int, default=0, help='job id')
parser.add_argument('--num-jobs', type=int, default=1,
                    help='number of parallel jobs')
random.seed(0)
np.random.seed(0)


def main():
    global args
    args = parser.parse_args()
    args.batch_size = 10  # only segment one image for experiment

    model_dir = os.path.dirname(args.dir)
    core_config_path = os.path.join(model_dir, 'configs/core.config')

    core_config = CoreConfig()
    core_config.read(core_config_path)
    print('Using core configuration from {}'.format(core_config_path))

    offset_list = core_config.offsets
    print("offsets are: {}".format(offset_list))

    testset = WaldoDataset(args.test_data, core_config, args.train_image_size)
    print('Total samples in the test set: {0}'.format(len(testset)))

    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

    segment_dir = args.dir
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)
    segment(dataloader, segment_dir, core_config)


def segment(dataloader, segment_dir, core_config):
    tot = 0
    rle_dir = os.path.join(segment_dir, 'rle')
    img_dir = os.path.join(segment_dir, 'img')
    if not os.path.exists(rle_dir):
        os.makedirs(rle_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    num_classes = core_config.num_classes
    offset_list = core_config.offsets

    img, class_pred, adj_pred = next(iter(dataloader))

##    print("{}".format(class_pred[0].detach().numpy().shape))
##    print("{}".format(adj_pred[0].detach().numpy().shape))
    """
    if args.object_merge_factor is None:
        args.object_merge_factor = 1.0 / len(offset_list)
        segmenter_opts = SegmenterOptions(same_different_bias=args.same_different_bias,
                                          object_merge_factor=args.object_merge_factor,
                                          merge_logprob_bias=args.merge_logprob_bias)
    start = time.time()   
    seg = ObjectSegmenter(class_pred[0].detach().numpy(),
                          adj_pred[0].detach().numpy(),
                          num_classes, offset_list,
                          segmenter_opts)
    mask_pred, object_class = seg.run_segmentation()
    end = time.time()
    tot = tot + end - start
    
    """
    if args.object_merge_factor is None:
        args.object_merge_factor = 1.0 / len(offset_list)

    epsilon = np.finfo(np.float32).eps
    class_pred_in = class_pred[0].detach().numpy().astype(np.float32).clip(epsilon, 1.0 - epsilon)
    adj_pred_in = adj_pred[0].detach().numpy().astype(np.float32).clip(epsilon, 1.0 - epsilon)

    offset_array = np.array(offset_list).astype(np.int32)
    
    mask_pred = np.zeros((class_pred[0].detach().numpy().shape[1],
                          class_pred[0].detach().numpy().shape[2])).astype(np.int32)
    object_class_pred = np.zeros((1,class_pred[0].detach().numpy().shape[1] *
                            class_pred[0].detach().numpy().shape[2])).astype(np.int32)
    #start = time.time()
    cseg.run_segmentation(class_pred_in, adj_pred_in,
                          num_classes,
                          offset_array,
                          mask_pred,
                          object_class_pred,
                          args.same_different_bias,
                          args.object_merge_factor,
                          args.merge_logprob_bias)
    #end = time.time()
    #tot = tot + end - start
    object_class = []
    for i in range(object_class_pred.shape[1] - 1):
        if object_class_pred[0, i] == -1: break
        object_class.append(object_class_pred[0, i])
    
    image_with_mask = {}
    img = np.moveaxis(img[0].detach().numpy(), 0, -1)
    image_with_mask['img'] = img
    image_with_mask['mask'] = mask_pred
    image_with_mask['object_class'] = object_class
    visual_mask = visualize_mask(image_with_mask, core_config)[
        'img_with_mask']
    scipy.misc.imsave('{}/oracle.png'.format(img_dir), visual_mask)

    rles = list(mask_to_rles(mask_pred))
    segment_rle_file = '{}/oracle.rle'.format(rle_dir)
    with open(segment_rle_file, 'w') as fh:
        for obj in rles:
            obj_str = ' '.join(str(n) for n in obj)
            fh.write(obj_str)
            fh.write('\n')
    print("{0}".format(tot))

##    mask_rle_file = '{}/oracle.img'.format(rle_dir)
##    with open(mask_rle_file, 'w') as fh:
##        np.savetxt(mask_rle_file, mask_pred, fmt="%d", delimiter=" ")



def rle_encoding(x):
    """ This function accepts a binary mask x of size (height, width) and 
        return its run-length encoding. run-length encoding will encode the
        binary mask as pairs of values that each pair contains a start position
        and its run length. Note that the pixels in a 2-dim mask are one-indexed,
        from top to bottom, then left to right. It follows the requirement
        from dsb2018 : "https://www.kaggle.com/c/data-science-bowl-2018#evaluation"  
        e.g. if x = [0 0 0 
                     1 1 1 
                     0 1 1 ], it will return [2 1 5 2 8 2]
    """
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def mask_to_rles(x):
    for i in range(1, x.max() + 1):
        yield rle_encoding((x == i).astype(int))


if __name__ == '__main__':
    main()
