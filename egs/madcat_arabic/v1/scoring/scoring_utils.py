#!/usr/bin/env python3

from shapely.geometry.polygon import Polygon
import numpy as np
from PIL import Image


def _evaluate_mask_image(mask_ref_arr, mask_hyp_arr, iou_threshold):
    """Given reference mask and hypothesis mask, returns iou matrix
        and precision, recall score. It requires same value for to represent an
        object in a mask.
        Returns
        -------
        a dict that contains:
        precision: (hypothesis matched)/(total hypothesis)
        recall: (hypothesis matched)/(total reference)
        pairs: list of matching hypothesis and reference pairs
        """

    # replace 0 and get all unique values in the mask
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
    iou_mat = np.zeros([len(mask_val_ref), len(mask_val_hyp)])

    # compute iou value
    for ref_index in range(len(mask_val_ref)):
        for hyp_index in range(len(mask_val_hyp)):
            val_ref = mask_val_ref[ref_index]
            val_hyp = mask_val_hyp[hyp_index]

            temp_img_ref = Image.new('L', (mask_ref_arr.shape[1], mask_ref_arr.shape[0]), 0)
            pixels = np.where(mask_ref_arr == val_ref,
                              mask_ref_arr, temp_img_ref)
            ref_bool_arr = np.array(pixels, dtype=bool)

            temp_img_hyp = Image.new('L', (mask_hyp_arr.shape[1], mask_hyp_arr.shape[0]), 0)
            pixels = np.where(mask_hyp_arr == val_hyp,
                              mask_hyp_arr, temp_img_hyp)
            hyp_bool_arr = np.array(pixels, dtype=bool)

            overlap = hyp_bool_arr * ref_bool_arr
            union = hyp_bool_arr + ref_bool_arr

            iou_mat[ref_index, hyp_index] = overlap.sum() / float(union.sum())

    # update score and get stats if iou value above threshold
    num_ref = len(mask_val_ref)
    num_hyp = len(mask_val_hyp)
    per_sample_metrics = get_stats(iou_mat, iou_threshold, num_hyp, num_ref)

    return per_sample_metrics


def _evaluate_text_file(ref_rect_list, hyp_rect_list, iou_threshold):
    """Given reference file and hypothesis file, returns iou matrix
    and precision, recall score. It requires reference and hypothesis
    file to contain a rectangle in each line. A rectangle is described
    by 8 values (x1,y1,x2,y2,x3,y3,x4,y4)
    Returns
    -------
    a dict that contains:
    precision: (hypothesis matched)/(total hypothesis)
    recall: (hypothesis matched)/(total reference)
    pairs: list of matching hypothesis and reference pairs
    """

    # get all polygons present in the file
    ref_polygons, ref_pol_points = _get_polygons(ref_rect_list)
    hyp_polygons, hyp_pol_points = _get_polygons(hyp_rect_list)

    # compute iou value
    iou_mat = np.zeros([len(ref_polygons), len(hyp_polygons)])
    for ref_index in range(len(ref_polygons)):
        for hyp_index in range(len(hyp_polygons)):
            polygon_ref = ref_polygons[ref_index]
            polygon_hyp = hyp_polygons[hyp_index]
            iou_mat[ref_index, hyp_index] = _get_intersection_over_union(polygon_hyp, polygon_ref)

    # update score and get stats if iou value above threshold
    num_ref = len(ref_polygons)
    num_hyp = len(hyp_polygons)
    per_sample_metrics = get_stats(iou_mat, iou_threshold, num_hyp, num_ref)

    return per_sample_metrics


def get_stats(iou_mat, iou_threshold, num_hyp, num_ref):
    """ Given iou matrix, it returns the precision and recall score
    based on the threshold
    """
    per_sample_metrics = dict()
    hyp_matched = 0
    pairs = []
    ref_rect_mat = np.zeros(num_ref, np.int8)
    hyp_rect_mat = np.zeros(num_hyp, np.int8)

    for ref_index in range(num_ref):
        for hyp_index in range(num_hyp):
            if ref_rect_mat[ref_index] == 0 and hyp_rect_mat[hyp_index] == 0:
                if iou_mat[ref_index, hyp_index] > iou_threshold:
                    ref_rect_mat[ref_index] = 1
                    hyp_rect_mat[hyp_index] = 1
                    hyp_matched += 1
                    pairs.append({'reference_data': ref_index, 'det': hyp_index})


    # compute precision and recall value
    if num_ref == 0:
        recall = float(1)
        precision = float(0) if num_hyp > 0 else float(1)
    else:
        recall = float(hyp_matched) / num_ref
        precision = 0 if num_hyp == 0 else float(hyp_matched) / num_hyp

    per_sample_metrics['precision'] = precision
    per_sample_metrics['recall'] = recall
    per_sample_metrics['pairs'] = pairs

    return per_sample_metrics


def _get_polygons(rect_list):
    """Given a file, it returns all the polygons
    present in the file.
    """

    polygons = []
    polygon_points = []
    for n in range(len(rect_list)):
        points = rect_list[n]
        polygon = _polygon_from_points(points)
        polygons.append(polygon)
        polygon_points.append(points)
    return polygons, polygon_points


def _get_union(hyp, ref):
    """Given two polygons it returns area of union.
    """
    area_a = hyp.area
    area_b = ref.area
    return area_a + area_b - _get_intersection(hyp, ref)


def _get_intersection_over_union(hyp, ref):
    """Given two polygons it returns the IOU value.
    IOU value is the ratio between the area of the intersection
    of the two polygons divided by the area of their union.
    """
    try:
        intersection = _get_intersection(hyp, ref)
        union = _get_union(hyp, ref)
        return intersection / union
    except:
        return 0


def _get_intersection(hyp, ref):
    """Given two polygons it returns area of
    intersection.
    """
    p_int = hyp & ref
    area = p_int.area
    return area


def _polygon_from_points(points):
    """Given a rectangle described by 8 values (x1,y1,x2,y2,x3,y3,x4,y4),
    returns a polygon object from shapely library
    """
    res_boxes = np.empty([1, 8], dtype='int32')
    res_boxes[0, 0] = int(points[0])
    res_boxes[0, 4] = int(points[1])
    res_boxes[0, 1] = int(points[2])
    res_boxes[0, 5] = int(points[3])
    res_boxes[0, 2] = int(points[4])
    res_boxes[0, 6] = int(points[5])
    res_boxes[0, 3] = int(points[6])
    res_boxes[0, 7] = int(points[7])
    point_mat = res_boxes[0].reshape([2, 4]).T
    return Polygon(point_mat)


def get_score(ref, hyp, iou_threshold, if_eval_text_file=True):

    if if_eval_text_file:
        return _evaluate_text_file(ref, hyp, iou_threshold)
    else:
        return _evaluate_mask_image(ref, hyp, iou_threshold)

