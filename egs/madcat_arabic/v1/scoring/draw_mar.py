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
parser.add_argument('--head', type=int, default=10,
                    help='determines number images to draw the MAR. If value set to negative then processes all images.')
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
    gt_mar = os.path.join(args.ground_truth_dir, 'mar_orig_dim.txt')
    pred_mar = os.path.join(args.predicted_dir, 'mar_orig_dim.txt')
    output = os.path.join(args.predicted_dir, 'img_orig')
    count = 0
    with open(gt_mar) as f1, open(pred_mar) as f2:
        for gt, pred in zip(f1, f2):
            if count >= args.head and args.head >= 0:
                break
            gt = gt.strip()
            pred = pred.strip()
            img_id = gt.split(' ')[0].split('$')[0]
            gt_points = gt.split(' ')[1]
            pred_points = pred.split(' ')[1]
            gt_list = [[int(y) for y in x.split(',')[:-1]] for x in gt_points.split(';')[:-1]]
            pred_list = [[int(y) for y in x.split(',')[:-1]] for x in pred_points.split(';')[:-1]]

            img_arr = np.load(args.ground_truth_dir + '/orig_img/' + img_id + '.orig_img.npy')
            img = Image.fromarray(img_arr)
            img = draw_rect(img, gt_list, 'BLACK')
            img = draw_rect(img, pred_list, 'RED')
            scipy.misc.imsave('{}/{}_orig.png'.format(output, img_id), img)
            count = count + 1


if __name__ == '__main__':
    main()
