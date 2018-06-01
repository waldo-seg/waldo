#!/usr/bin/env python3

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
    mean_ar, stat_dict = get_mean_avg_score(threshold_list)
    write_stats_to_file(mean_ar, stat_dict)


def get_mean_avg_score(threshold_list):
    mean_ar = 0
    stat_dict = {}
    for threshold in threshold_list:
        mean_recall = 0
        img_count = 0
        for img_ref_path, img_hyp_path in zip(glob(args.reference + "/*.mask.npy"),
                                              glob(args.hypothesis + "/*.mask.npy")):
            img_count += 1
            ref_arr = np.load(img_ref_path)
            hyp_arr = np.load(img_hyp_path)
            image_id = os.path.basename(img_ref_path).split('.mask.npy')[0]
            recall = get_score(ref_arr, hyp_arr, threshold)['recall']
            mean_recall += recall
            if image_id not in stat_dict.keys():
                stat_dict[image_id] = dict()
            stat_dict[image_id][threshold] = recall
        mean_recall /= img_count
        print("For threshold: {} Mean recall: {}".format(threshold, mean_recall))
        mean_ar += mean_recall
    mean_ar /= len(threshold_list)
    print("Mean average recall: {}".format(mean_ar))
    return mean_ar, stat_dict


def write_stats_to_file(mean_ar, stat_dict):
    with open(args.result, 'w') as fh:
        fh.write('Mean Average Recall: {}\n'.format(mean_ar))
        fh.write('ImageID\tThreshold\tRecall\n')
        for image_id in stat_dict.keys():
            for threshold in stat_dict[image_id].keys():
                recall = stat_dict[image_id][threshold]
                fh.write('{}\t{}\t{}\n'.format(image_id, threshold, recall))


if __name__ == '__main__':
      main()
