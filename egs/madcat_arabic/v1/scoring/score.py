#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

# This script performs scoring based mask image or rectangle coordinates.
# It calculates mean average recall and mean average precision for a given
# threshold list. It calls utility functions from scoring_utils.py

import argparse
import os
from scoring_utils import get_score
from glob import glob
import numpy as np

parser = argparse.ArgumentParser(
    description='scoring script for text localization')
parser.add_argument('--reference', type=str, required=True,
                    help='reference directory of test data, contains np array')
parser.add_argument('--hypothesis', type=str, required=True,
                    help='hypothesis directory of test data, contains np array')
parser.add_argument('--result', type=str, required=True,
                    help='the file to store final statistical results')
args = parser.parse_args()


def main():
    threshold_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ref_image_rect_dict = read_rect_coordinates(args.reference)
    hyp_image_rect_dict = read_rect_coordinates(args.hypothesis)
    mean_ap, mean_ar, stat_dict = get_mean_avg_score_from_rect_coordinates(
                   threshold_list, ref_image_rect_dict, hyp_image_rect_dict)
    write_stats_to_file(mean_ap, mean_ar, stat_dict)


def get_mean_avg_score_from_rect_coordinates(threshold_list, ref_image_rect_dict,
                                             hyp_image_rect_dict):
    """
        Given the threshold list, it returns a tuple (mean_ap, mean_ar, stat_dict): 
        mean average precision, mean average recall and statistic dictionary
        input
        -----
        threshold_list [float]: list of threshold values. MAP and MAR
        are calculated for this threshold list.
        ref_image_rect_dict: dict([[int]]): dict of a list of list, for
          each image_id it contains a list of rectangle and a rectangle
          is a list containing 8 integer values.
        hyp_image_rect_dict : dict([[int]]): dict of a list of list, for
          each image_id it contains a list of rectangle and a rectangle
          is a list containing 8 integer values.
        return
        -----
        mean_ap (float): mean average precision over threshold list.
         will satsify 0 <= mean_ap <= 1
        mean_ar (float): mean average recall over threshold list.
         will satsify 0 <= mean_ar <= 1
        stat_dict dict(dict): contains precision and recall value for each
         image for each threshold
    """
    mean_ar = 0
    mean_ap = 0
    stat_dict = {}
    for threshold in threshold_list:
        mean_recall = 0
        mean_precision = 0
        img_count = 0
        for image_id in ref_image_rect_dict.keys():
            img_count += 1
            ref_rect_coord_list = ref_image_rect_dict[image_id]
            hyp_rect_coord_list = hyp_image_rect_dict[image_id]
            score = get_score(ref_rect_coord_list, hyp_rect_coord_list, threshold)
            precision = score['precision']
            recall = score['recall']
            mean_precision += precision
            mean_recall += recall
            precision_recall = str(precision) + " " + str(recall)
            if image_id not in stat_dict.keys():
                stat_dict[image_id] = dict()
            stat_dict[image_id][threshold] = precision_recall
        mean_precision /= img_count
        mean_recall /= img_count
        print("For threshold: {} Mean precision: {:0.3f} Mean recall: {:0.3f}".format(
                                     threshold, mean_precision, mean_recall))
        mean_ap += mean_precision
        mean_ar += mean_recall
    mean_ap /= len(threshold_list)
    mean_ar /= len(threshold_list)
    print("Mean average precision: {} Mean average recall: {}".
                                      format(mean_ap, mean_ar))
    return mean_ap, mean_ar, stat_dict


def get_mean_avg_score_from_mask_image(threshold_list):
    """
        Given the threshold list, returns a tuple (mean_ap, mean_ar, stat_dict):
        mean average precision, mean average recall and statistic dictionary
        input
        -----
        threshold_list [float]: list of threshold values. MAP and MAR
        are calculated for this threshold list.
        return
        -----
        mean_ap (float): mean average precision over threshold list.
         will satsify 0 <= mean_ap <= 1
        mean_ar (float): mean average recall over threshold list.
         will satsify 0 <= mean_ar <= 1
        stat_dict dict(dict): contains precision and recall value for each
         image for each threshold
    """
    mean_ar = 0
    mean_ap = 0
    stat_dict = {}
    for threshold in threshold_list:
        mean_recall = 0
        mean_precision = 0
        img_count = 0
        for img_ref_path, img_hyp_path in zip(glob(args.reference + "/*.mask.npy"),
                                              glob(args.hypothesis + "/*.mask.npy")):
            img_count += 1
            ref_arr = np.load(img_ref_path)
            hyp_arr = np.load(img_hyp_path)
            image_id = os.path.basename(img_ref_path).split('.mask.npy')[0]
            score = get_score(ref_arr, hyp_arr, threshold)
            precision = score['precision']
            recall = score['recall']
            mean_precision += precision
            mean_recall += recall
            precision_recall = str(precision) + " " + str(recall)
            if image_id not in stat_dict.keys():
                stat_dict[image_id] = dict()
            stat_dict[image_id][threshold] = precision_recall
        mean_precision /= img_count
        mean_recall /= img_count
        print("For threshold: {} Mean precision: {:0.3f} Mean recall: {:0.3f}".format(
                                     threshold, mean_precision, mean_recall))
        mean_ap += mean_precision
        mean_ar += mean_recall
    mean_ap /= len(threshold_list)
    mean_ar /= len(threshold_list)
    print("Mean average precision: {} Mean average recall: {}".
                                      format(mean_ap, mean_ar))
    return mean_ap, mean_ar, stat_dict


def write_stats_to_file(mean_ap, mean_ar, stat_dict):
    """ Given mean average precision, mean average recall
        and statistic dictionary, it writes image_id, threshold,
        precision and recall value in args.result text file.
       input
       -----
       mean_ap (float): mean average precision over threshold list.
       will satsify 0 <= mean_ap <= 1.
       mean_ar (float): mean average recall over threshold list.
       will satsify 0 <= mean_ar <= 1
       stat_dict dict(dict): contains precision and recall value for each
       image for each threshold
    """
    with open(args.result, 'w') as fh:
        fh.write('Mean Average Precision: {}\n'.format(mean_ap))
        fh.write('Mean Average Recall: {}\n'.format(mean_ar))
        fh.write('ImageID  Threshold  Recall\n')
        for image_id in stat_dict.keys():
            for threshold in stat_dict[image_id].keys():
                recall = stat_dict[image_id][threshold]
                fh.write('{}  {}  {}\n'.format(image_id, threshold, recall))
    print('Saved to {}'.format(args.result))


def read_rect_coordinates(file_name):
    """ Given the file name, it reads mask_id and rectangle
        coordinates from the file. It finally returns a image_rect_dict.
        A file should contain mask_id and the co-ordinates of the 
        minimum area rectangle that covers the mask with that mask id, 
        in the form of a counter-clockwise list of points. A mar is 
        described by 8 values (h1,w1,h2,w2,h3,w3,h4,w4), in the format:
          <mask-id> h1,w1,h2,w2,h3,w3,h4,w4
        for example:
          HYT_ARB_20070103.0066_4_LDC0061 25,179,15,178,16,70,26,71
        return
        ------
        image_rect_dict : dict([[int]]): dict of a list of list, for
          each image_id it contains a list of rectangle and a rectangle
          is a list containing 8 integer values (h1,w1,h2,w2,h3,w3,h4,w4)
    """
    image_rect_dict = {}
    with open(file_name) as f:
        for line in f:
            line_vect = line.strip().split(' ')
            image_id = line_vect[0]
            rect_coordinates = line_vect[1].split(',')[:-1]
            if image_id not in image_rect_dict.keys():
                image_rect_dict[image_id] = list()
            image_rect_dict[image_id].append(rect_coordinates)
    return image_rect_dict


if __name__ == '__main__':
    main()
