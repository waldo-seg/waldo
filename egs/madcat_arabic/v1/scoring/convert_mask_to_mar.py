#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

import numpy as np
import argparse
import os
import sys
from glob import glob
from waldo.data_manipulation import get_mar_from_mask

parser = argparse.ArgumentParser(description='converts np array to rle format')
parser.add_argument('--indir', type=str, required=True,
                    help='directory of data, contains np array')
parser.add_argument('--outdir', type=str, required=True,
                    help='the file to store final statistical results')
args = parser.parse_args()


def write_rects_to_file(out_dir, mar_dict):
    """ Given an output directory and mar_dictionary, it writes mask_id and
    mar coordinates to mar.txt file
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
                fh.write('{} {}\n'.format(mask_id, point_str))
    print('Saved to {}'.format(txt_path))


def main():
    mar_dict = dict()
    for mask_path in glob(args.indir+"/*.mask.npy"):
        mask_arr = np.load(mask_path)
        mask_id = os.path.basename(mask_path).split('.mask.npy')[0]
        mask_mar_list = list(get_mar_from_mask(mask_arr))
        mar_dict[mask_id] = mask_mar_list
    write_rects_to_file(args.outdir, mar_dict)


if __name__ == '__main__':
    main()
