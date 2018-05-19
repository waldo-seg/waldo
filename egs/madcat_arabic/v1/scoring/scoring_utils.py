from collections import namedtuple
from shapely.geometry.polygon import Polygon
import numpy as np


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def get_score(reference_data, hypothesis_data):

    _validate_data(reference_data, hypothesis_data)
    score = _evaluate_data(reference_data, hypothesis_data)

    return score


def _evaluate_data(reference_data, hypothesis_data):
    per_sample_metrics = {}
    matched_sum = 0
    gt = reference_data
    subm = hypothesis_data

    num_global_care_gt = 0
    num_global_care_det = 0

    for result_file in gt:
        gt_file = gt
        subm_file = subm
        det_matched = 0
        iou_mat = np.empty([1, 1])

        gt_pols = []
        det_pols = []

        gt_pol_points = []
        det_pol_points = []

        pairs = []
        det_matched_nums = []

        pointlist = _get_pointlist(gt_file)
        for n in range(len(pointlist)):
            points = pointlist[n]
            gtRect = Rectangle(*points)
            gt_pol = _rectangle_to_polygon(gtRect)
            gt_pol = _polygon_from_points(points)
            gt_pols.append(gt_pol)
            gt_pol_points.append(points)

        if result_file in subm:

            pointlist = _get_pointlist(subm_file)
            for n in range(len(pointlist)):
                points = pointlist[n]
                detRect = Rectangle(*points)
                det_pol = _rectangle_to_polygon(detRect)
                det_pol = _polygon_from_points(points)
                det_pols.append(det_pol)
                det_pol_points.append(points)

            if len(gt_pols) > 0 and len(det_pols) > 0:
                # Calculate IoU and precision matrixs
                outputShape = [len(gt_pols), len(det_pols)]
                iou_mat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gt_pols), np.int8)
                detRectMat = np.zeros(len(det_pols), np.int8)
                for gt_num in range(len(gt_pols)):
                    for det_num in range(len(det_pols)):
                        p_g = gt_pols[gt_num]
                        p_d = det_pols[det_num]
                        iou_mat[gt_num, det_num] = _get_intersection_over_union(p_d, p_g)

            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    if gtRectMat[gt_num] == 0 and detRectMat[det_num] == 0:
                        if iou_mat[gt_num, det_num] > 0.5:
                            gtRectMat[gt_num] = 1
                            detRectMat[det_num] = 1
                            det_matched += 1
                            pairs.append({'gt': gt_num, 'det': det_num})
                            det_matched_nums.append(det_num)

        num_gt_care = len(gt_pols)
        num_det_care = len(det_pols)
        if num_gt_care == 0:
            recall = float(1)
            precision = float(0) if num_det_care > 0 else float(1)
        else:
            recall = float(det_matched) / num_gt_care
            precision = 0 if num_det_care == 0 else float(det_matched) / num_det_care

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        matched_sum += det_matched
        num_global_care_gt += num_gt_care
        num_global_care_det += num_det_care

        per_sample_metrics[result_file] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iou_mat': [] if len(det_pols) > 100 else iou_mat.tolist(),
            'gt_pol_points': gt_pol_points,
            'det_pol_points': det_pol_points
        }

    # Compute MAP and MAR

    method_recall = 0 if num_global_care_gt == 0 else float(matched_sum) / num_global_care_gt
    method_precision = 0 if num_global_care_det == 0 else float(matched_sum) / num_global_care_det
    method_hmean = 0 if method_recall + method_precision == 0 else 2 * method_recall * method_precision / (
    method_recall + method_precision)

    method_metrics = {'precision': method_precision, 'recall': method_recall, 'hmean': method_hmean}

    resDict = {'calculated': True, 'Message': '', 'method': method_metrics, 'per_sample': per_sample_metrics}

    return resDict


def _get_pointlist(gt_file):
    points = []
    return points


def _validate_data(reference_data, hypothesis_data):
    points = []
    _validate_clockwise_points(points)
    return


def _validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """
    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
        [int(points[0]), int(points[1])],
        [int(points[2]), int(points[3])],
        [int(points[4]), int(points[5])],
        [int(points[6]), int(points[7])]
    ]
    edge = [
        (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
        (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
        (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
        (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3]
    if summatory > 0:
        raise Exception(
            "Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. "
            "Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the "
            "standard one, with the image origin at the upper left, the X axis extending to the right and Y axis "
            "extending downwards.")


def _polygon_from_points(points):
    """
    Returns a Polygon object from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    point_mat = resBoxes[0].reshape([2, 4]).T
    return Polygon(point_mat)


def _rectangle_to_polygon(rect):
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(rect.xmin)
    resBoxes[0, 4] = int(rect.ymax)
    resBoxes[0, 1] = int(rect.xmin)
    resBoxes[0, 5] = int(rect.ymin)
    resBoxes[0, 2] = int(rect.xmax)
    resBoxes[0, 6] = int(rect.ymin)
    resBoxes[0, 3] = int(rect.xmax)
    resBoxes[0, 7] = int(rect.ymax)

    point_mat = resBoxes[0].reshape([2, 4]).T

    return Polygon(point_mat)


def _get_union(p_d, p_g):
    area_a = p_d.area()
    area_b = p_g.area()
    return area_a + area_b - _get_intersection(p_d, p_g)


def _get_intersection_over_union(p_d, p_g):
    try:
        return _get_intersection(p_d, p_g) / _get_union(p_d, p_g)
    except:
        return 0


def _get_intersection(p_d, p_g):
    p_int = p_d & p_g
    if len(p_int) == 0:
        return 0
    return p_int.area()


