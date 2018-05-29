#!/usr/bin/env python3

import argparse
import os
from scoring_utils import get_score
from glob import glob
import numpy as np

parser = argparse.ArgumentParser(
    description='scoring script for text localization')
parser.add_argument('hypothesis', type=str,
                    help='hypothesis directory of test data')
parser.add_argument('reference', type=str,
                    help='reference directory of test data')
args = parser.parse_args()


def main():
    iou_threshold = 0.5
    precision = 0
    count = 0

    for img_ref_path, img_hyp_path in zip(glob(args.reference+"*.png"), glob(args.hypothesis+"*.png")):

        ref_arr = np.load(img_ref_path)
        hyp_arr = np.load(img_hyp_path)
        score = get_score(ref_arr, hyp_arr, iou_threshold)
        precision += score['precision']
        count += 1

    precision /= count

    print("Total Precision: {}".format(precision))


if __name__ == '__main__':
      main()

