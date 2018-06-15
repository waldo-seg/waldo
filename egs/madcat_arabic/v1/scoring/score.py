#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

# This script performs scoring based mask image or rectangle coordinates.
# It calculates mean average recall and mean average precision for a given
# threshold list. It calls utility functions from scoring_utils.py

import argparse
import os
from scoring_utils import get_score, get_mar_transcription_mapping
from glob import glob
import numpy as np

parser = argparse.ArgumentParser(
    description='scoring script for text localization')
parser.add_argument('reference', type=str,
                    help='reference directory of test data, contains np array')
parser.add_argument('hypothesis', type=str,
                    help='hypothesis directory of test data, contains np array')
parser.add_argument('result', type=str,
                    help='the file to store final statistical results')
parser.add_argument('--mar-text-mapping', type=str, default=None,
                    help="If not none, map hypothesis mar with the transciptions."
                        "A hypothesis box is mapped with the transcription "
                        "of reference box that had the largest IoU overlap."
                        "The variable will provide the path of the reference " 
                        "file containing mapping between mar and text")
parser.add_argument("--score-mar", action="store_true",
                   help="If true, score after finding the minimum area rectangle"
                        " derived from the object mask. If false, score based on" 
                        " object mask without further processing.")
args = parser.parse_args()

def main():
    threshold_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    if args.score_mar:
        ref_dict = read_rect_coordinates(args.reference)
        hyp_dict = read_rect_coordinates(args.hypothesis)
    else:
        ref_dict, hyp_dict = get_filenames_from_directory()

    mean_ap, mean_ar, stat_dict = get_mean_avg_scores(
            threshold_list, ref_dict, hyp_dict)

    write_stats_to_file(mean_ap, mean_ar, stat_dict)
    if args.mar_text_mapping:
        mapping_file = os.path.join(args.result, 'mar_transcription_mapping.txt')
        with open(mapping_file, 'w') as mapping_fh:
            ref_dict = read_rect_coordinates_and_transcription(args.mar_text_mapping)
            hyp_dict = read_rect_coordinates(args.hypothesis)
            for image_id in hyp_dict:
                ref_rect_transcription_list = ref_dict[image_id]
                for hyp_rect in hyp_dict[image_id]:
                    ref_rect_transcription, best_index = get_mar_transcription_mapping(
                        ref_rect_transcription_list, hyp_rect)
                    mapping_fh.write('{}  {}  {}  {}\n'.format(image_id, hyp_rect, ref_rect_transcription, best_index))


def get_mean_avg_scores(threshold_list, ref_dict, hyp_dict):
    """
        Given the threshold list, it returns a tuple (mean_ap, mean_ar, stat_dict): 
        mean average precision, mean average recall and statistic dictionary
        input
        -----
        If args.score_mar == true, then
          ref_dict : dict([[int]]): dict of a list of list, for
          each image_id it contains a list of rectangle and a rectangle
          is a list containing 8 integer values
          hyp_dict : dict([[int]]): dict of a list of list, for
          each image_id it contains a list of rectangle and a rectangle
          is a list containing 8 integer values.
        else
          ref_dict : dict(str): a dict of file basename and file path, for
                     all files in the reference directory
          hyp_dict : dict(str): a dict of file basename and file path, for
                     all files in the hypothesis directory

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
        for image_id in ref_dict.keys():
            img_count += 1
            ref_data = ref_dict[image_id]
            hyp_data = hyp_dict[image_id]
            if not args.score_mar:
                ref_data = np.load(ref_data)
                hyp_data = np.load(hyp_data)
            score = get_score(ref_data, hyp_data, threshold, args.score_mar)
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
    result_file = os.path.join(args.result, 'scoring_result.txt')
    with open(result_file, 'w') as result_fh:
        result_fh.write('Mean Average Precision: {}\n'.format(mean_ap))
        result_fh.write('Mean Average Recall: {}\n'.format(mean_ar))
        result_fh.write('ImageID  Threshold  Recall\n')
        for image_id in stat_dict.keys():
            for threshold in stat_dict[image_id].keys():
                recall = stat_dict[image_id][threshold]
                result_fh.write('{}  {}  {}\n'.format(image_id, threshold, recall))
    print('Saved to {}'.format(result_file))


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


def read_rect_coordinates_and_transcription(file_name):
    """ Given the file name, it reads mask_id, rectangle
        coordinates and transcription from the file. It finally
        returns a image_rect_dict. A file should contain mask_id and
        the co-ordinates of the mar that covers the mask with that mask id,
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
            image_id = line_vect[0][:-5]
            rect_coordinates = line_vect[1].split(',')
            transcription = " ".join(line_vect[2:])
            if image_id not in image_rect_dict.keys():
                image_rect_dict[image_id] = list()
            image_rect_dict[image_id].append((rect_coordinates, transcription))
    return image_rect_dict


def get_filenames_from_directory():
    """ Given the hypothesis and reference directory name, it returns
    two dicts containing file name of each directory respectively. It
    checks if both directory contains same files names.
    To do: add partial scoring option similar to kaldi.
    return
    ------
    ref_file_dict : dict(str): a dict of file basename and file path, for
                    all files in the reference directory
    hyp_file_dict : dict(str): a dict of file basename and file path, for
                    all files in the hypothesis directory
    """

    ref_file_dict = dict()
    hyp_file_dict = dict()
    for img_ref_path, img_hyp_path in zip(glob(args.reference + "/*.mask.npy"),
                                          glob(args.hypothesis + "/*.mask.npy")):

        ref_id = os.path.basename(img_ref_path).split('.mask.npy')[0]
        hyp_id = os.path.basename(img_hyp_path).split('.mask.npy')[0]
        ref_file_dict[ref_id] = img_ref_path
        hyp_file_dict[hyp_id] = img_hyp_path

    assert len(ref_file_dict) == len(hyp_file_dict)

    for file_id in ref_file_dict.keys():
        if file_id not in hyp_file_dict.keys():
            raise Exception("mask flie (np array): {} missing in reference directory".format(file_id))

    return ref_file_dict, hyp_file_dict


if __name__ == '__main__':
    main()

