#!/usr/bin/env python3

import numpy as np
import csv
import argparse
import os
import sys
from glob import glob

parser = argparse.ArgumentParser(description='converts np array to rle format')
parser.add_argument('--indir', type=str, required=True,
                    help='directory of data, contains np array')
parser.add_argument('--outdir', type=str, required=True,
                    help='the file to store final statistical results')
parser.add_argument('--csv', type=str, default='sub-dsbowl2018.csv',
                    help='Csv filename as the final submission file')
args = parser.parse_args()


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


def main():
    rle_dir = os.path.join(args.outdir, 'rle')
    if not os.path.exists(rle_dir):
        os.makedirs(rle_dir)

    for mask_path in glob(args.indir+"/*.mask.npy"):
        mask_arr = np.load(mask_path)
        mask_id = os.path.basename(mask_path).split('.mask.npy')[0]
        rles = list(mask_to_rles(mask_arr))
        rle_file = '{}/{}.rle'.format(rle_dir, mask_id)
        with open(rle_file, 'w') as fh:
            for obj in rles:
                obj_str = ' '.join(str(n) for n in obj)
                fh.write(obj_str)
                fh.write('\n')

    make_submission(args.outdir, args.csv)

if __name__ == '__main__':
    main()

