#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

import argparse
import os
import sys
import numpy as np
from glob import glob
from waldo.data_manipulation import get_mar_from_mask
from waldo.mar_utils import dilate_polygon

parser = argparse.ArgumentParser(description='converts np array to rle format')
parser.add_argument('--indir', type=str, required=True,
                    help='directory of data, contains np array')
parser.add_argument('--outdir', type=str, required=True,
                    help='the file to store final statistical results')
parser.add_argument('--cur-size', type=int, default=None,
                    help='the current scale of the images')
parser.add_argument('--sizedir', type=str, default=None,
                    help='the directory that contains the original sizes')
args = parser.parse_args()


def write_rects_to_file(out_dir, mar_dict):
    """ Given an output directory and mar_dictionary, it writes mask_id and
    the co-ordinates of the minimum area rectangle that covers the mask 
    with that mask id, in the form of a counter-clockwise list of points. 
    A mar is described by 8 values (h1,w1,h2,w2,h3,w3,h4,w4), in the format:
      <mask-id> h1,w1,h2,w2,h3,w3,h4,w4
    for example:
      HYT_ARB_20070103.0066_4_LDC0061 25,179,15,178,16,70,26,71
    input
    -----
    out_dir (str): path of output directory
    mar_dict (dict): mar dictionary containing mask_id and list of list of coordinates
    """
    txt_path = os.path.join(out_dir, 'mar.txt')
    with open(txt_path, 'w') as fh:
        for mask_id in mar_dict.keys():
            mask_mar_list = mar_dict[mask_id]
            for mar in mask_mar_list:
                point_str = str()
                for point in mar:
                    point_str = point_str + str(point[0]) + ',' + str(point[1]) + ','
                min_h = min(mar[0][0], mar[1][0], mar[2][0], mar[3][0])
                min_w = min(mar[0][1], mar[1][1], mar[2][1], mar[3][1])
                max_h = max(mar[0][0], mar[1][0], mar[2][0], mar[3][0])
                max_w = max(mar[0][1], mar[1][1], mar[2][1], mar[3][1])
                mask_line_id = mask_id + '_' + str(min_h) + '_' + str(min_w) + '_' + str(max_h) + '_' + str(max_w)
                fh.write('{} {}\n'.format(mask_line_id, point_str))
    print('Saved to {}'.format(txt_path))

def write_rects_to_file_orig_dim(out_dir, mar_dict):
    """ Given an output directory and mar_dictionary, it writes mask_id and
    the co-ordinates of the minimum area rectangle that covers the mask
    with that mask id, in the form of a counter-clockwise list of points.
    A mar is described by 8 values (h1,w1,h2,w2,h3,w3,h4,w4), in the format:
      <mask-id> h1,w1,h2,w2,h3,w3,h4,w4
    for example:
      HYT_ARB_20070103.0066_4_LDC0061 25,179,15,178,16,70,26,71
    input
    -----
    out_dir (str): path of output directory
    mar_dict (dict): mar dictionary containing mask_id and list of list of coordinates
    """
    txt_path = os.path.join(out_dir, 'mar.txt')
    txt_path2 = os.path.join(out_dir, 'mar_orig_dim.txt')
    mar_orig_fh = open(txt_path2, 'w')
    with open(txt_path, 'w') as fh:
        for mask_id in mar_dict.keys():
            mask_mar_list = mar_dict[mask_id]
            dim_arr = np.load(args.sizedir + '/' + mask_id + '.orig_dim.npy')
            scale = (1.0 * np.amax(dim_arr)) / args.cur_size
            for mar in mask_mar_list:
                mar = dilate_polygon(mar, 1.5)
                point_str = str()
                point_str_scaled = str()
                for point in mar:
                    point_str = point_str + str(point[0]) + ',' + str(point[1]) + ','
                    point_str_scaled = point_str_scaled + str(int(point[0]*scale)) + ',' + str(int(point[1]*scale)) + ','
                min_h = min(mar[0][0], mar[1][0], mar[2][0], mar[3][0])
                min_w = min(mar[0][1], mar[1][1], mar[2][1], mar[3][1])
                max_h = max(mar[0][0], mar[1][0], mar[2][0], mar[3][0])
                max_w = max(mar[0][1], mar[1][1], mar[2][1], mar[3][1])
                mask_line_id = mask_id + '_' + str(min_h) + '_' + str(min_w) + '_' + str(max_h) + '_' + str(max_w)
                mask_line_id_slaled = mask_id + '_' + str(min_h*scale) + '_' + str(min_w*scale) + '_' + str(max_h*scale) + '_' + str(max_w*scale)
                fh.write('{} {}\n'.format(mask_line_id, point_str))
                mar_orig_fh.write('{} {}\n'.format(mask_line_id_slaled, point_str_scaled))
    print('Saved to {}'.format(txt_path))

def main():
    mar_dict = dict()
    for mask_path in sorted(glob(args.indir+"/*.mask.npy")):
        mask_arr = np.load(mask_path)
        mask_id = os.path.basename(mask_path).split('.mask.npy')[0]
        mask_mar_list = list(get_mar_from_mask(mask_arr))
        mar_dict[mask_id] = mask_mar_list
    if args.cur_size and args.sizedir:
        write_rects_to_file_orig_dim(args.outdir, mar_dict)
    else:
        write_rects_to_file(args.outdir, mar_dict)


if __name__ == '__main__':
    main()
