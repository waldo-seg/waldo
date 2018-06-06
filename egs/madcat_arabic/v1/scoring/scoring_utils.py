#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

# It contains utility functions for scoring. These functions are called from score.py

from shapely.geometry.polygon import Polygon
import numpy as np
from PIL import Image


def _evaluate_mask_image(mask_ref_arr, mask_hyp_arr, iou_threshold):
    """Given reference mask and hypothesis mask, it returns precision
    and recall score. It requires same values of object pixels in the mask.
    input
    -----
    mask_ref_arr (np array): ref array, contains same value for each line MAR
    mask_hyp_arr (np array): hyp array, contains same value for each line MAR
    iou_threshold (float): should be between 0 and 1, it decides if a match is
     a good match or not.
    Returns
    -------
    a dict that contains:
    precision: (hypothesis matched)/(total hypothesis)
    recall: (hypothesis matched)/(total reference)
    pairs: list of matching hypothesis and reference pairs
    """

    # replace 0 with a value between 0,255 assuming number
    # of objects is less than 255
    # 0 values is used to create a temporary blank image
    mask_val_ref = np.unique(mask_ref_arr)
    mask_val_hyp = np.unique(mask_hyp_arr)
    for val in range(0,255):
        if val not in mask_val_ref:
            mask_ref_arr[mask_ref_arr == 0] = val
            break

    for val in range(0,255):
        if val not in mask_val_hyp:
            mask_hyp_arr[mask_hyp_arr == 0] = val
            break

    mask_val_ref = np.unique(mask_ref_arr)
    mask_val_hyp = np.unique(mask_hyp_arr)
    num_ref = len(mask_val_ref)
    num_hyp = len(mask_val_hyp)
    iou_score = np.zeros([num_ref, num_hyp])

    # compute intersection over union value
    # for each ref and hyp object pair. Store
    # the value in iou_score matrix
    # iou_score[ref_index, hyp_index] is the iou
    # of reference object (ref_index) and hypothesis object (hyp_index).
    for ref_index in range(num_ref):
        for hyp_index in range(num_hyp):
            val_ref = mask_val_ref[ref_index]
            val_hyp = mask_val_hyp[hyp_index]

            # for a given object, create a boolen image with object pixels as true and
            # background pixels as false
            temp_img_ref = Image.new('L', (mask_ref_arr.shape[1], mask_ref_arr.shape[0]), 0)
            pixels = np.where(mask_ref_arr == val_ref,
                              mask_ref_arr, temp_img_ref)
            ref_bool_arr = np.array(pixels, dtype=bool)

            temp_img_hyp = Image.new('L', (mask_hyp_arr.shape[1], mask_hyp_arr.shape[0]), 0)
            pixels = np.where(mask_hyp_arr == val_hyp,
                              mask_hyp_arr, temp_img_hyp)
            hyp_bool_arr = np.array(pixels, dtype=bool)

            # calculate intersection and union values
            intersection = hyp_bool_arr * ref_bool_arr
            union = hyp_bool_arr + ref_bool_arr

            iou_score[ref_index, hyp_index] = intersection.sum() / float(union.sum())

    # get stats for a given iou threshold value
    score = get_stats(iou_score, iou_threshold)

    return score


def _evaluate_text_file(ref_rect_list, hyp_rect_list, iou_threshold):
    """Given reference rectangle list and hypothesis rectangle list, it
    returns precision and recall score. It requires reference and hypothesis
    rectangle list to contain rectangle in each line. A rectangle is described
    by 8 values (x1,y1,x2,y2,x3,y3,x4,y4)
    input
    -----
    ref_rect_list [[int]]: contains a list of list, contains a list of rectangle
    and a rectangle is a list containing 8 integer values.
    hyp_rect_list [[int]]: contains a list of list, contains a list of rectangle
    and a rectangle is a list containing 8 integer values.
    iou_threshold (float): should be between 0 and 1, it decides if a match is
     a good match or not.
    Returns
    -------
    a dict that contains:
    precision: (hypothesis matched)/(total hypothesis)
    recall: (hypothesis matched)/(total reference)
    pairs: list of matching hypothesis and reference pairs
    """

    # get all polygons present in the file
    ref_polygons = _get_rect_in_shapely_format(ref_rect_list)
    hyp_polygons = _get_rect_in_shapely_format(hyp_rect_list)
    num_ref = len(ref_polygons)
    num_hyp = len(hyp_polygons)

    # compute intersection over union value
    # for each ref and hyp object pair. Store
    # the value in iou_score matrix
    # iou_score[ref_index, hyp_index] is the iou
    # of reference object (ref_index) and hypothesis object (hyp_index).
    iou_score = np.zeros([num_ref, num_hyp])
    for ref_index in range(num_ref):
        for hyp_index in range(num_hyp):
            polygon_ref = ref_polygons[ref_index]
            polygon_hyp = hyp_polygons[hyp_index]
            iou_score[ref_index, hyp_index] = _get_intersection_over_union(polygon_hyp, polygon_ref)

    # get stats for a given iou threshold value
    score = get_stats(iou_score, iou_threshold)

    return score


def get_stats(iou_score, iou_threshold):
    """
    Given iou score for each ref hyp pair, it returns the precision
    and recall score.
    input
    -----
    iou_score [num_ref, num_hyp]: iou score between ref and hyp pair,
    all values should be between 0 and 1
    iou_threshold (float): should be between 0 and 1, it decides if a match is
     a good match or not.
    return
    -----
    a dict that contains:
    precision: (hypothesis matched)/(total hypothesis)
    recall: (hypothesis matched)/(total reference)
    pairs: list of matching hypothesis and reference pairs
    """
    hyp_matched = 0
    pairs = []
    num_ref = iou_score.shape[0]
    num_hyp = iou_score.shape[1]
    if_ref_object_matched = np.zeros(num_ref, np.int8)
    if_hyp_object_matched = np.zeros(num_hyp, np.int8)

    for ref_index in range(num_ref):
        for hyp_index in range(num_hyp):
            if if_ref_object_matched[ref_index] == 0 and if_hyp_object_matched[hyp_index] == 0:
                if iou_score[ref_index, hyp_index] > iou_threshold:
                    if_ref_object_matched[ref_index] = 1
                    if_hyp_object_matched[hyp_index] = 1
                    hyp_matched += 1
                    pairs.append({'reference_data': ref_index, 'det': hyp_index})


    # compute precision and recall value
    if num_ref == 0:
        recall = float(1)
        precision = float(0) if num_hyp > 0 else float(1)
    else:
        recall = float(hyp_matched) / num_ref
        precision = 0 if num_hyp == 0 else float(hyp_matched) / num_hyp

    score = dict()
    score['precision'] = precision
    score['recall'] = recall
    score['pairs'] = pairs

    return score


def _get_intersection_over_union(hyp_rect, ref_rect):
    """Given two rectangles (hyp and ref) in shapely format,
    it returns the IOU value. IOU value is the ratio between
    the area of the intersection of the two polygons divided
    by the area of their union.
    Returns
    -------
    iou_val: (float)
    """
    try:
        rect_intersection = hyp_rect & ref_rect
        intersection_area = rect_intersection.area
        union_area = hyp_rect.area + ref_rect.area - intersection_area
        iou_val = float(intersection_area) / union_area
        return iou_val
    except:
        return 0


def _get_rect_in_shapely_format(rect_list):
    """
    Given a rectangle list, it convert and returns the rectangle
    in shapely format. Shapely library is used to calculate intersection
    and union area of two rectangles.
    input
    -----
    rect_list [[int]]: contains a list of list, contains a list of rectangle
    and a rectangle is a list containing 8 integer values. These values are
    (x1,y1,x2,y2,x3,y3,x4,y4)
    return
    ------
    rect_list: list of rectangle in shapely format
    """

    rect_shapely = []
    for n in range(len(rect_list)):
        points = rect_list[n]
        rect_coordinate = np.empty([1, 8], dtype='int32')
        rect_coordinate[0, 0] = int(points[0])
        rect_coordinate[0, 4] = int(points[1])
        rect_coordinate[0, 1] = int(points[2])
        rect_coordinate[0, 5] = int(points[3])
        rect_coordinate[0, 2] = int(points[4])
        rect_coordinate[0, 6] = int(points[5])
        rect_coordinate[0, 3] = int(points[6])
        rect_coordinate[0, 7] = int(points[7])
        rect_coordinate_reshaped = rect_coordinate[0].reshape([2, 4]).T
        rect = Polygon(rect_coordinate_reshaped)
        rect_shapely.append(rect)
    return rect_shapely


def get_score(ref, hyp, iou_threshold, if_eval_text_file=True):
    """
    input
    -----
    If if_eval_text_file == true, then
      ref : [[int]]: contains a list of list, contains a list of rectangle
        and a rectangle is a list containing 8 integer values.
      hyp : [[int]]: contains a list of list, contains a list of rectangle
        and a rectangle is a list containing 8 integer values.
    else
      ref : (np array): ref array, contains same value for each line MAR
      hyp : (np array): hyp array, contains same value for each line MAR

    iou_threshold (float): should be between 0 and 1, it decides if a match is
      a good match or not.
    if_eval_text_file: bool, checks if evaluation should be based on text file or
      mask image
    Returns
    -------
    a dict that contains:
    precision: (hypothesis matched)/(total hypothesis)
    recall: (hypothesis matched)/(total reference)
    pairs: list of matching hypothesis and reference pairs
    """
    if if_eval_text_file:
        return _evaluate_text_file(ref, hyp, iou_threshold)
    else:
        return _evaluate_mask_image(ref, hyp, iou_threshold)

