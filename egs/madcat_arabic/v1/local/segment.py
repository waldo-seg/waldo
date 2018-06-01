#!/usr/bin/env python3

import torch
import argparse
import os
import sys
import random
import numpy as np
import scipy.misc
from models.Unet import UNet
from waldo.segmenter import ObjectSegmenter, SegmenterOptions
from waldo.core_config import CoreConfig
from waldo.data_visualization import visualize_mask
from unet_config import UnetConfig
import scipy
import csv
from skimage.transform import resize
from waldo.data_io import WaldoTestset

parser = argparse.ArgumentParser(description='Pytorch MADCAT Arabic setup')
parser.add_argument('--test-data', type=str, required=True,
                    help='Path to test images to be segmented')
parser.add_argument('--dir', type=str, required=True,
                    help='Directory to store segmentation results. '
                    'It is assumed that <dir> is a sub-directory of '
                    'the model directory.')
parser.add_argument('--model', type=str, default='model_best.pth.tar',
                    help='Name of the model file to use for segmenting.')
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
    args.batch_size = 1  # only segment one image for experiment

    model_dir = os.path.dirname(args.dir)
    core_config_path = os.path.join(model_dir, 'configs/core.config')
    unet_config_path = os.path.join(model_dir, 'configs/unet.config')

    core_config = CoreConfig()
    core_config.read(core_config_path)
    print('Using core configuration from {}'.format(core_config_path))

    # loading Unet configuration
    unet_config = UnetConfig()
    unet_config.read(unet_config_path, args.train_image_size)
    print('Using unet configuration from {}'.format(unet_config_path))

    offset_list = core_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    num_classes = core_config.num_classes
    num_colors = core_config.num_colors
    num_offsets = len(core_config.offsets)
    # model configurations from unet config
    start_filters = unet_config.start_filters
    up_mode = unet_config.up_mode
    merge_mode = unet_config.merge_mode
    depth = unet_config.depth

    model = UNet(num_classes, num_offsets,
                 in_channels=num_colors, depth=depth,
                 start_filts=start_filters,
                 up_mode=up_mode,
                 merge_mode=merge_mode)

    model_path = os.path.join(model_dir, args.model)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded.")
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    testset = WaldoTestset(args.test_data, args.train_image_size,
                           job=args.job, num_jobs=args.num_jobs)
    print('Total samples in the test set: {0}'.format(len(testset)))

    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

    segment_dir = args.dir
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)

    segment(dataloader, segment_dir, model, core_config)
    make_submission(segment_dir, args.csv)

def segment(dataloader, segment_dir, model, core_config):
    model.eval()  # convert the model into evaluation mode
    img_dir = os.path.join(segment_dir, 'img')
    rle_dir = os.path.join(segment_dir, 'rle')
    mask_dir = os.path.join(segment_dir, 'mask_pred')
    if not os.path.exists(rle_dir):
        os.makedirs(rle_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    exist_ids = next(os.walk(img_dir))[2]

    num_classes = core_config.num_classes
    offset_list = core_config.offsets

    for i, (img, size, id) in enumerate(dataloader):
        id = id[0]  # tuple to str
        if id + '.png' in exist_ids:
            continue
        original_height, original_width = size[0].item(), size[1].item()
        with torch.no_grad():
            output = model(img)
            # class_pred = (output[:, :num_classes, :, :] + 0.001) * 0.999
            # adj_pred = (output[:, num_classes:, :, :] + 0.001) * 0.999
            class_pred = output[:, :num_classes, :, :]
            adj_pred = output[:, num_classes:, :, :]

        if args.object_merge_factor is None:
            args.object_merge_factor = 1.0 / len(offset_list)
            segmenter_opts = SegmenterOptions(same_different_bias=args.same_different_bias,
                                              object_merge_factor=args.object_merge_factor)
        seg = ObjectSegmenter(class_pred[0].detach().numpy(),
                              adj_pred[0].detach().numpy(),
                              num_classes, offset_list,
                              segmenter_opts)
        mask_pred, object_class = seg.run_segmentation()
        #mask_pred = resize(mask_pred, (original_height, original_width),
        #                   order=0, preserve_range=True).astype(int)

        image_with_mask = {}
        img = np.moveaxis(img[0].detach().numpy(), 0, -1)
        #img = resize(img, (original_height, original_width),
        #             preserve_range=True)
        image_with_mask['img'] = img
        image_with_mask['mask'] = mask_pred
        image_with_mask['object_class'] = object_class
        visual_mask = visualize_mask(image_with_mask, core_config)[
            'img_with_mask']
        scipy.misc.imsave('{}/{}.png'.format(img_dir, id), visual_mask)
        #scipy.misc.imsave('{}/{}.png'.format(mask_dir, id), mask_pred)
        filename = mask_dir + '/' + id + '.' + 'mask' + '.npy'
        np.save(filename, mask_pred)
        rles = list(mask_to_rles(mask_pred))
        segment_rle_file = '{}/{}.rle'.format(rle_dir, id)
        with open(segment_rle_file, 'w') as fh:
            for obj in rles:
                obj_str = ' '.join(str(n) for n in obj)
                fh.write(obj_str)
                fh.write('\n')


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

def make_submission(segment_dir, csvname):
    rle_dir = os.path.join(segment_dir, 'rle')
    ids = next(os.walk(rle_dir))[2]
    csv_path = os.path.join(segment_dir, csvname)
    with open(csv_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['ImageId', 'EncodedPixels'])
        for id in ids:
            with open(os.path.join(rle_dir, id), 'r') as rlefile:
                image_id = id.split('.')[0]
                for line in rlefile:
                    line = line.strip()
                    csv_writer.writerow([image_id] + [line])

    print('Saved to {}'.format(csv_path))


if __name__ == '__main__':
    main()

