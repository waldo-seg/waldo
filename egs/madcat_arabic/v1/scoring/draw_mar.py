#!/usr/bin/env python3

# Copyright 2018 Chun-Chieh Chang

import argparse
import os
import scipy.misc
import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description='Draws bounding boxes on original images')
parser.add_argument('ground_truth_dir', type=str,
                    help='directory with original image and original MAR')
parser.add_argument('predicted_dir', type=str,
                    help='directory with predicted MAR')
args = parser.parse_args()

def draw_rect(img, rect, color):
    img_draw = ImageDraw.Draw(img)
    for poly in rect:
        count = 0
        for point in poly:
            if count % 2 == 1:
                temp = poly[count - 1]
                poly[count - 1] = poly[count]
                poly[count] = temp
            count = count + 1
        img_draw.polygon(poly, outline=color)
    return img

def main():
    gt_mar = os.path.join(args.ground_truth_dir, 'mar_orig.txt')
    pred_mar = os.path.join(args.predicted_dir, 'mar_orig.txt')
    output = os.path.join(args.predicted_dir, 'img_orig')
    with open(gt_mar) as f1, open(pred_mar) as f2:
        for gt, pred in zip(f1, f2):
            gt = gt.strip()
            pred = pred.strip()
            img_id = gt.split(' ')[0]
            gt_points = gt.split(' ')[1]
            pred_points = pred.split(' ')[1]
            gt_list = [[int(y) for y in x.split(',')[:-1]] for x in gt_points.split(';')[:-1]]
            pred_list = [[int(y) for y in x.split(',')[:-1]] for x in pred_points.split(';')[:-1]]

            img_arr = np.load(args.ground_truth_dir + '/orig_img/' + img_id + '.orig_img.npy')
            img = Image.fromarray(img_arr)
            img = draw_rect(img, gt_list, 'BLACK')
            img = draw_rect(img, pred_list, 'RED')
            scipy.misc.imsave('{}/{}_orig.png'.format(output, img_id), img)


if __name__ == '__main__':
    main()
