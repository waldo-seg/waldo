#!/usr/bin/env python3

# Copyright   2018 Ashish Arora
# Copyright   2018 Chun-Chieh Chang
# Apache 2.0

# minimum bounding box script is originally from
#https://github.com/BebeSparkelSparkel/MinimumBoundingBox

# dilate and erode script is inspired by
# https://stackoverflow.com/a/3897471

""" It is a collection of utility functions that finds minimum area rectangle (MAR).
 Given the list of points, get_mar function finds a MAR that contains all the
 points and have minimum area. The obtained MAR (not necessarily be vertically
 or horizontally aligned).
"""

from math import atan2, cos, sin, pi, sqrt
from collections import namedtuple
import numpy as np
from scipy.spatial import ConvexHull

"""
bounding_box is a named tuple which contains:
             area (float): area of the rectangle
             length_parallel (float): length of the side that is parallel to unit_vector
             length_orthogonal (float): length of the side that is orthogonal to unit_vector
             rectangle_center(int, int): coordinates of the rectangle center
             unit_vector (float, float): direction of the length_parallel side.
             unit_vector_angle (float): angle of the unit vector to be in radians.
             corner_points [(float, float)]: set that contains the corners of the rectangle
"""
bounding_box_tuple = namedtuple('bounding_box_tuple', 'area '
                                        'length_parallel '
                                        'length_orthogonal '
                                        'rectangle_center '
                                        'unit_vector '
                                        'unit_vector_angle '
                                        'corner_points '
                         )



def _unit_vector(pt0, pt1):
    """ Given two points pt0 and pt1, return a unit vector that
        points in the direction of pt0 to pt1.
    Returns
    -------
    (float, float): unit vector
    """
    dis_0_to_1 = sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def _orthogonal_vector(vector):
    """ Given a vector, returns a orthogonal/perpendicular vector of equal length.
    Returns
    ------
    (float, float): A vector that points in the direction orthogonal to vector.
    """
    return -1 * vector[1], vector[0]


def _bounding_area(index, hull):
    """ Given index location in an array and convex hull, it gets two points
        hull[index] and hull[index+1]. From these two points, it returns a named
        tuple that mainly contains area of the box that bounds the hull. This
        bounding box orintation is same as the orientation of the lines formed
        by the point hull[index] and hull[index+1].
    Returns
    -------
    a named tuple that contains:
    area: area of the rectangle
    length_parallel: length of the side that is parallel to unit_vector
    length_orthogonal: length of the side that is orthogonal to unit_vector
    rectangle_center: coordinates of the rectangle center
    unit_vector: direction of the length_parallel side.
    (it's orthogonal vector can be found with the orthogonal_vector function)
    """
    unit_vector_p = _unit_vector(hull[index], hull[index + 1])
    unit_vector_o = _orthogonal_vector(unit_vector_p)

    dis_p = tuple(np.dot(unit_vector_p, pt) for pt in hull)
    dis_o = tuple(np.dot(unit_vector_o, pt) for pt in hull)

    min_p = min(dis_p)
    min_o = min(dis_o)
    len_p = max(dis_p) - min_p
    len_o = max(dis_o) - min_o

    return {'area': len_p * len_o,
            'length_parallel': len_p,
            'length_orthogonal': len_o,
            'rectangle_center': (min_p + len_p / 2, min_o + len_o / 2),
            'unit_vector': unit_vector_p,
            }


def _to_xy_coordinates(unit_vector_angle, point):
    """ Given angle from horizontal axis and a point from origin,
        returns converted unit vector coordinates in x, y coordinates.
        angle of unit vector should be in radians.
    Returns
    ------
    (float, float): converted x,y coordinate of the unit vector.
    """
    angle_orthogonal = unit_vector_angle + pi / 2
    return point[0] * cos(unit_vector_angle) + point[1] * cos(angle_orthogonal), \
           point[0] * sin(unit_vector_angle) + point[1] * sin(angle_orthogonal)


def _rotate_points(center_of_rotation, angle, points):
    """ Rotates a point cloud around the center_of_rotation point by angle
    input
    -----
    center_of_rotation (float, float): angle of unit vector to be in radians.
    angle (float): angle of rotation to be in radians.
    points [(float, float)]: Points to be a list or tuple of points. Points to be rotated.
    Returns
    ------
    [(int, int)]: Rotated points around center of rotation by angle
    """
    rot_points = []
    ang = []
    for pt in points:
        diff = tuple([pt[d] - center_of_rotation[d] for d in range(2)])
        diff_angle = atan2(diff[1], diff[0]) + angle
        ang.append(diff_angle)
        diff_length = sqrt(sum([d**2 for d in diff]))
        rot_points.append((int(center_of_rotation[0] + diff_length * cos(diff_angle)),
                           int(center_of_rotation[1] + diff_length * sin(diff_angle))))

    return rot_points


def _rectangle_corners(rectangle):
    """ Given rectangle center and its inclination, returns the corner
        locations of the rectangle.
    Returns
    ------
    [(float, float)]: 4 corner points of rectangle.
    """
    corner_points = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1 * i1):
            corner_points.append((rectangle['rectangle_center'][0] + i1 * rectangle['length_parallel'],
                            rectangle['rectangle_center'][1] + i2 * rectangle['length_orthogonal']))

    return _rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)


def _get_mask_points(img_arr):
    """ Given an image numpy array, it returns all the points in each object
    in the image. Assuming background have maximum number of points, it will 
    remove background object points.
    input
    -----
    img_arr : (np array): image array, contains same value for each object
    Returns
    -------
    a dict that contains:
    object_id: [[int]] contains a list of list, a list of points
    and a point is a list containing 2 integer values.
    """
    img_unique_val = np.unique(img_arr)
    max_point_object_id = -1
    max_num_points = -1
    masks_point_dict = dict()
    for mask_id in img_unique_val:
        points_location = np.where(img_arr == mask_id)
        min_height = min(points_location[0])
        max_height = max(points_location[0])
        min_width = min(points_location[1])
        max_width = max(points_location[1])
        # not a 2D data for convex hull function
        if (max_height - min_height) <= 2 or (max_width - min_width) <= 2:
            continue

        mask_points = list(zip(points_location[0], points_location[1]))
        mask_points = list(set(mask_points))  # unique points in the mask
        if len(mask_points) <= 2:
            continue

        masks_point_dict[mask_id] = mask_points
        if len(mask_points) > max_num_points:
            max_num_points = len(mask_points)
            max_point_object_id = mask_id

    # assuming background have maximum number of points
    if max_point_object_id != -1:
        del masks_point_dict[max_point_object_id]

    return masks_point_dict


def dilate_polygon(points, amount_increase):
    """ Increases size of polygon given as a list of tuples. Assumes points in polygon are given in CCW
    """
    expanded_points = []
    for index, point in enumerate(points):
        prev_point = points[(index - 1) % len(points)]
        next_point = points[(index + 1) % len(points)]
        prev_edge = np.subtract(point, prev_point)
        next_edge = np.subtract(next_point, point)
        
        prev_normal = ((1 * prev_edge[1]), (-1 * prev_edge[0]))
        prev_normal = np.divide(prev_normal, np.linalg.norm(prev_normal))
        next_normal = ((1 * next_edge[1]), (-1 * next_edge[0]))
        next_normal = np.divide(next_normal, np.linalg.norm(next_normal))

        bisect = np.add(prev_normal, next_normal)
        bisect = np.divide(bisect, np.linalg.norm(bisect))
        
        cos_theta = np.dot(next_normal, bisect)
        hyp = amount_increase / cos_theta
        
        new_point = np.around(point + hyp * bisect)
        new_point = new_point.astype(int)
        new_point = new_point.tolist()
        expanded_points.append(new_point)
    return expanded_points


def erode_polygon(points, amount_increase):
    """ Increases size of polygon given as a list of tuples. Assumes points in polygon are given in CCW
    """
    expanded_points = []
    for index, point in enumerate(points):
        prev_point = points[(index - 1) % len(points)]
        next_point = points[(index + 1) % len(points)]
        prev_edge = np.subtract(point, prev_point)
        next_edge = np.subtract(next_point, point)

        prev_normal = ((-1 * prev_edge[1]), (1 * prev_edge[0]))
        prev_normal = np.divide(prev_normal, np.linalg.norm(prev_normal))
        next_normal = ((-1 * next_edge[1]), (1 * next_edge[0]))
        next_normal = np.divide(next_normal, np.linalg.norm(next_normal))

        bisect = np.add(prev_normal, next_normal)
        bisect = np.divide(bisect, np.linalg.norm(bisect))

        cos_theta = np.dot(next_normal, bisect)
        hyp = amount_increase / cos_theta

        new_point = np.around(point + hyp * bisect)
        new_point = new_point.astype(int)
        new_point = new_point.tolist()
        expanded_points.append(new_point)
    return expanded_points


def get_rectangles_from_mask(img_arr):
    """ Given an image numpy array, it returns a minimum area rectangle that will
    contain the mask. It will not necessarily be vertically or horizontally
     aligned.
    input
    -----
    img_arr : (np array): image array, contains same value for each mask
    Returns
    -------
    list((int, int)): 4 corner points of rectangle.
    """
    masks_point_dict = _get_mask_points(img_arr)
    mar_list = list()
    for object_id in masks_point_dict.keys():
        mask_points = masks_point_dict[object_id]
        mask_points = tuple(mask_points)
        hull_ordered = [mask_points[index] for index in ConvexHull(mask_points).vertices]
        hull_ordered.append(hull_ordered[0])  # making it cyclic, now first and last point are same

        # not a rectangle
        if len(hull_ordered) < 5:
            continue

        hull_ordered = tuple(hull_ordered)
        min_rectangle = _bounding_area(0, hull_ordered)
        for i in range(1, len(hull_ordered) - 1):
            rectangle = _bounding_area(i, hull_ordered)
            if rectangle['area'] < min_rectangle['area']:
                min_rectangle = rectangle

        min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1],
                                                   min_rectangle['unit_vector'][0])

        min_rectangle['rectangle_center'] = _to_xy_coordinates(min_rectangle['unit_vector_angle'],
                                                              min_rectangle['rectangle_center'])
        rect_corners = _rectangle_corners(min_rectangle)

        rect_corners = tuple(rect_corners)
        points_ordered = [rect_corners[index] for index in ConvexHull(rect_corners).vertices]
        mar_list.append(points_ordered)
    return mar_list


def get_mar(polygon):
    """ Given a list of points, returns a minimum area rectangle that will
    contain all points. It will not necessarily be vertically or horizontally
     aligned.
    Returns
    -------
    list((int, int)): 4 corner points of rectangle.
    """
    polygon = tuple(polygon)
    hull_ordered = [polygon[index] for index in ConvexHull(polygon).vertices]
    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)
    min_rectangle = _bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered) - 1):
        rectangle = _bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = _to_xy_coordinates(min_rectangle['unit_vector_angle'],
                                                           min_rectangle['rectangle_center'])
    points_list = _rectangle_corners(min_rectangle)
    return points_list


def get_rectangle(polygon):
    """ Given a list of points, returns a minimum area rectangle that will
    contain all points. It will not necessarily be vertically or horizontally
     aligned.
    Returns
    -------
    list((int, int)): 4 corner points of rectangle.
    """
    polygon = tuple(polygon)
    hull_ordered = [polygon[index] for index in ConvexHull(polygon).vertices]
    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)
    min_rectangle = _bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered) - 1):
        rectangle = _bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = _to_xy_coordinates(min_rectangle['unit_vector_angle'],
                                                           min_rectangle['rectangle_center'])
    return bounding_box_tuple(
        area=min_rectangle['area'],
        length_parallel=min_rectangle['length_parallel'],
        length_orthogonal=min_rectangle['length_orthogonal'],
        rectangle_center=min_rectangle['rectangle_center'],
        unit_vector=min_rectangle['unit_vector'],
        unit_vector_angle=min_rectangle['unit_vector_angle'],
        corner_points=set(_rectangle_corners(min_rectangle))
    )

