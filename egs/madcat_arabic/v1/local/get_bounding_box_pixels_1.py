import sys
import argparse
import os
import xml.dom.minidom as minidom
import numpy as np
from math import atan2, cos, sin, pi, degrees, sqrt
from collections import namedtuple

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arrow, Circle
from scipy.spatial import ConvexHull
from PIL import Image
from scipy.misc import toimage

bounding_box_tuple = namedtuple('bounding_box_tuple', 'area '
                                                      'length_parallel '
                                                      'length_orthogonal '
                                                      'rectangle_center '
                                                      'unit_vector '
                                                      'unit_vector_angle '
                                                      'corner_points'
                                )

def unit_vector(pt0, pt1):
    dis_0_to_1 = sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1

def orthogonal_vector(vector):
    return -1 * vector[1], vector[0]

def bounding_area(index, hull):
    unit_vector_p = unit_vector(hull[index], hull[index + 1])
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
    angle_orthogonal = unit_vector_angle + pi / 2
    return point[0] * cos(unit_vector_angle) + point[1] * cos(angle_orthogonal), \
           point[0] * sin(unit_vector_angle) + point[1] * sin(angle_orthogonal)

def rotate_points(center_of_rotation, angle, points):
    rot_points = []
    ang = []
    for pt in points:
        diff = tuple([pt[d] - center_of_rotation[d] for d in range(2)])
        diff_angle = atan2(diff[1], diff[0]) + angle
        ang.append(diff_angle)
        diff_length = sqrt(sum([d ** 2 for d in diff]))
        rot_points.append((center_of_rotation[0] + diff_length * cos(diff_angle),
                           center_of_rotation[1] + diff_length * sin(diff_angle)))

    return rot_points

def rectangle_corners(rectangle):
    corner_points = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1 * i1):
            corner_points.append((rectangle['rectangle_center'][0] + i1 * rectangle['length_parallel'],
                                  rectangle['rectangle_center'][1] + i2 * rectangle['length_orthogonal']))

    return rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)

def minimum_bounding_box(points):
    if len(points) <= 2: raise ValueError('More than two points required.')

    hull_ordered = [points[index] for index in ConvexHull(points).vertices]
    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)

    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered) - 1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = to_xy_coordinates(min_rectangle['unit_vector_angle'],
                                                          min_rectangle['rectangle_center'])

    return bounding_box_tuple(
        area=min_rectangle['area'],
        length_parallel=min_rectangle['length_parallel'],
        length_orthogonal=min_rectangle['length_orthogonal'],
        rectangle_center=min_rectangle['rectangle_center'],
        unit_vector=min_rectangle['unit_vector'],
        unit_vector_angle=min_rectangle['unit_vector_angle'],
        corner_points=set(rectangle_corners(min_rectangle))
    )

def get_center(im):
    center_x = im.size[0] / 2
    center_y = im.size[1] / 2
    return int(center_x), int(center_y)

def pad_image(image):
    width_padding = 200
    height_padding = 200
    stitched_image = Image.new('RGB', (image.size[0] + 2 * width_padding, image.size[1] + 2 * height_padding), "white")
    width_offset = 200
    height_offset = 200
    stitched_image.paste(im=image, box=(width_offset, height_offset))
    return stitched_image

def update_minimum_bounding_box_input(bounding_box_input):
    width_padding = 200
    height_padding = 200
    updated_minimum_bounding_box_input = []
    for point in bounding_box_input:
        x, y = point
        new_x = x + width_padding
        new_y = y + height_padding
        word_coordinate = (new_x, new_y)
        updated_minimum_bounding_box_input.append(word_coordinate)

    return updated_minimum_bounding_box_input

def set_line_image_data(image, line_id, image_file_name):
    base_name = os.path.splitext(os.path.basename(image_file_name))[0]
    line_image_file_name = base_name + line_id + '.tif'
    image_path = os.path.join(data_path, 'lines', line_image_file_name)
    imgray = image.convert('L')
    imgray.save(image_path)

def get_horizontal_angle(unit_vector_angle):
    if unit_vector_angle > pi / 2 and unit_vector_angle <= pi:
        unit_vector_angle = unit_vector_angle - pi
    elif unit_vector_angle > -pi and unit_vector_angle < -pi / 2:
        unit_vector_angle = unit_vector_angle + pi

    return unit_vector_angle

def get_smaller_angle(bounding_box):
    unit_vector = bounding_box.unit_vector
    unit_vector_angle = bounding_box.unit_vector_angle
    ortho_vector = orthogonal_vector(unit_vector)
    ortho_vector_angle = atan2(ortho_vector[1], ortho_vector[0])

    unit_vector_angle_updated = get_horizontal_angle(unit_vector_angle)
    ortho_vector_angle_updated = get_horizontal_angle(ortho_vector_angle)

    if abs(unit_vector_angle_updated) < abs(ortho_vector_angle_updated):
        return unit_vector_angle_updated
    else:
        return ortho_vector_angle_updated

def rotate_rectangle_corners(bounding_box, center, if_opposite_direction=False):
    p1, p2, p3, p4 = bounding_box.corner_points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    center_x, center_y = center

    if if_opposite_direction:
        rotation_angle_in_rad = get_smaller_angle(bounding_box)
    else:
        rotation_angle_in_rad = -get_smaller_angle(bounding_box)

    x_dash_1 = (x1 - center_x) * cos(rotation_angle_in_rad) - (y1 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_2 = (x2 - center_x) * cos(rotation_angle_in_rad) - (y2 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_3 = (x3 - center_x) * cos(rotation_angle_in_rad) - (y3 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_4 = (x4 - center_x) * cos(rotation_angle_in_rad) - (y4 - center_y) * sin(rotation_angle_in_rad) + center_x

    y_dash_1 = (y1 - center_y) * cos(rotation_angle_in_rad) + (x1 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_2 = (y2 - center_y) * cos(rotation_angle_in_rad) + (x2 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_3 = (y3 - center_y) * cos(rotation_angle_in_rad) + (x3 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_4 = (y4 - center_y) * cos(rotation_angle_in_rad) + (x4 - center_x) * sin(rotation_angle_in_rad) + center_y

    return (x_dash_1, y_dash_1), (x_dash_2, y_dash_2), (x_dash_3, y_dash_3), (x_dash_4, y_dash_4)


def rotate_single_point(point, bounding_box, center, if_opposite_direction=False):
    x1, y1 = point
    center_x, center_y = center

    if if_opposite_direction:
        rotation_angle_in_rad = get_smaller_angle(bounding_box)
    else:
        rotation_angle_in_rad = -get_smaller_angle(bounding_box)

    x_dash_1 = (x1 - center_x) * cos(rotation_angle_in_rad) - (y1 - center_y) * sin(rotation_angle_in_rad) + center_x
    y_dash_1 = (y1 - center_y) * cos(rotation_angle_in_rad) + (x1 - center_x) * sin(rotation_angle_in_rad) + center_y
    return x_dash_1, y_dash_1

def get_line_images_from_page_image(image_file_name, madcat_file_path):
    im_wo_pad = Image.open(image_file_name)
    im = pad_image(im_wo_pad)
    doc = minidom.parse(madcat_file_path)
    zone = doc.getElementsByTagName('zone')
    for node in zone:
        id = node.getAttribute('id')
        if id != 'z1':
            continue
        token_image = node.getElementsByTagName('token-image')
        minimum_bounding_box_input = []
        for token_node in token_image:
            word_point = token_node.getElementsByTagName('point')
            for word_node in word_point:
                word_coordinate = (int(word_node.getAttribute('x')), int(word_node.getAttribute('y')))
                minimum_bounding_box_input.append(word_coordinate)

        updated_mbb_input = update_minimum_bounding_box_input(minimum_bounding_box_input)
        bounding_box = minimum_bounding_box(updated_mbb_input)

        gBBC1, gBBC2, gBBC3, gBBC4 = bounding_box.corner_points

        x1, y1 = gBBC1
        x2, y2 = gBBC2
        x3, y3 = gBBC3
        x4, y4 = gBBC4
        gBBCmin_x = int(min(x1, x2, x3, x4))
        gBBCmin_y = int(min(y1, y2, y3, y4))
        gBBCmax_x = int(max(x1, x2, x3, x4))
        gBBCmax_y = int(max(y1, y2, y3, y4))
        BBwidth_half_x = (gBBCmax_x - gBBCmin_x) / 2
        BBheight_half_y = (gBBCmax_y - gBBCmin_y) / 2

        rel_BBC1 = (x1 - gBBCmin_x, y1 - gBBCmin_y)
        rel_BBC2 = (x2 - gBBCmin_x, y2 - gBBCmin_y)
        rel_BBC3 = (x3 - gBBCmin_x, y3 - gBBCmin_y)
        rel_BBC4 = (x4 - gBBCmin_x, y4 - gBBCmin_y)

        rel_points = []
        rel_points.append(rel_BBC1)
        rel_points.append(rel_BBC2)
        rel_points.append(rel_BBC3)
        rel_points.append(rel_BBC4)
        cropped_bounding_box = bounding_box_tuple(bounding_box.area,
                                                  bounding_box.length_parallel,
                                                  bounding_box.length_orthogonal,
                                                  bounding_box.length_orthogonal,
                                                  bounding_box.unit_vector,
                                                  bounding_box.unit_vector_angle,
                                                  set(rel_points)
                                                  )
        (rot_x1, rot_y1), (rot_x2, rot_y2), (rot_x3, rot_y3), (rot_x4, rot_y4) = rotate_rectangle_corners(
            cropped_bounding_box, (BBwidth_half_x, BBheight_half_y))
        patches = []
        rot_BBCmin_x = int(min(rot_x1, rot_x2, rot_x3, rot_x4))
        rot_BBCmin_y = int(min(rot_y1, rot_y2, rot_y3, rot_y4))
        rot_BBCmax_x = int(max(rot_x1, rot_x2, rot_x3, rot_x4))
        rot_BBCmax_y = int(max(rot_y1, rot_y2, rot_y3, rot_y4))

        # width = rot_BBCmax_x - rot_BBCmin_x
        # height = rot_BBCmax_y - rot_BBCmin_y
        # rot_points = []
        # rot_points.append((rot_x1, rot_y1))
        # rot_points.append((rot_x2, rot_y2))
        # rot_points.append((rot_x3, rot_y3))
        # rot_points.append((rot_x4, rot_y4))
        # rot_points_old = []
        #
        # for x,y in rot_points:
        #     point = x, y
        #     x1, y1 = rotate_single_point(point, cropped_bounding_box, (BBwidth_half_x, BBheight_half_y), True)
        #     gBBC1_old = (x1 + gBBCmin_x, y1 + gBBCmin_y)
        #     rot_points_old.append(gBBC1_old)
        #     patches.append(Circle((list(gBBC1_old)), radius=5, color='red'))

        for x in range(rot_BBCmin_x,rot_BBCmax_x):
            for y in range(rot_BBCmin_y, rot_BBCmax_y):
                point = x, y
                x1, y1 = rotate_single_point(point, cropped_bounding_box, (BBwidth_half_x, BBheight_half_y), True)
                gBBC1_old = (x1 + gBBCmin_x, y1 + gBBCmin_y)
                print(gBBC1_old)
                patches.append(Circle((list(gBBC1_old)), radius=1, color='red'))

        ax.imshow(im)
        for p in patches:
            ax.add_patch(p)
        plt.show(fig)

### main ###
fig,ax = plt.subplots(1)
line_images_path = '/Users/ashisharora/madcat_ar'
data_path = '/Users/ashisharora/madcat_ar'
for file in os.listdir(os.path.join(data_path, 'images')):
    if file.endswith(".tif"):
        image_path = os.path.join(data_path, 'images', file)
        gedi_file_path = os.path.join(data_path, 'madcat', file)
        gedi_file_path = gedi_file_path.replace(".tif", ".madcat.xml")
        get_line_images_from_page_image(image_path, gedi_file_path)