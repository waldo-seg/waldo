#!/usr/bin/env python3

# minimum bounding box  is originally from
#https://github.com/BebeSparkelSparkel/MinimumBoundingBox
#https://startupnextdoor.com/computing-convex-hull-in-python/

""" It is a collection of utility functions that finds minimum area rectangle (mar).
 Given the list of points, get_mar functionn finds a mar that contains all the
 points. The obtained mar (not necessarily be vertically or horizontally
 aligned) will have smallest area.
"""

from math import atan2, cos, sin, pi, sqrt
from collections import namedtuple
import numpy as np

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


def get_orientation(origin, p1, p2):
    """ Given origin and two points, return the orientation of the Point p1 with
        regards to Point p2 using origin.
        Returns
        -------
        integer: Negative if p1 is clockwise of p2.
        """
    difference = (
        ((p2[0] - origin[0]) * (p1[1] - origin[1]))
        - ((p1[0] - origin[0]) * (p2[1] - origin[1]))
    )
    return difference


def compute_hull(points):
    """ Given input list of points, return a list of points that
        made up the convex hull.
        Returns
        -------
        [(float, float)]: convexhull points
        """
    hull_points = []
    start = points[0]
    min_x = start[0]
    for p in points[1:]:
        if p[0] < min_x:
            min_x = p[0]
            start = p

    point = start
    hull_points.append(start)

    far_point = None
    while far_point is not start:
        p1 = None
        for p in points:
            if p is point:
                continue
            else:
                p1 = p
                break

        far_point = p1

        for p2 in points:
            if p2 is point or p2 is p1:
                continue
            else:
                direction = get_orientation(point, far_point, p2)
                if direction > 0:
                    far_point = p2

        hull_points.append(far_point)
        point = far_point
    return hull_points


def unit_vector(pt0, pt1):
    """ Given two points pt0 and pt1, return a unit vector that
        points in the direction of pt0 to pt1.
    Returns
    -------
    (float, float): unit vector
    """
    dis_0_to_1 = sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    """ Given a vector, returns a orthogonal/perpendicular vector of equal length.
    Returns
    ------
    (float, float): A vector that points in the direction orthogonal to vector.
    """
    return -1 * vector[1], vector[0]


def bounding_area(index, hull):
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
    unit_vector_p = unit_vector(hull[index], hull[index+1])
    unit_vector_o = orthogonal_vector(unit_vector_p)

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


def to_xy_coordinates(unit_vector_angle, point):
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


def rotate_points(center_of_rotation, angle, points):
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


def rectangle_corners(rectangle):
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

    return rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)


def get_mar(polygon):
    """ Given a list of points, returns a minimum area rectangle that will
    contain all points. It will not necessarily be vertically or horizontally
     aligned.
    Returns
    -------
    list((int, int)): 4 corner points of rectangle.
    """
    hull_ordered = compute_hull(list(polygon))
    hull_ordered = tuple(hull_ordered)
    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered) - 1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = to_xy_coordinates(min_rectangle['unit_vector_angle'],
                                                          min_rectangle['rectangle_center'])
    points_list = rectangle_corners(min_rectangle)
    return points_list