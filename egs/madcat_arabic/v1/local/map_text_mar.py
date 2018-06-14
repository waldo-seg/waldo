#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Ashish Arora)
# Apache 2.0

""" This script prepares the training, validation and test data for MADCAT Arabic in a pytorch fashion
"""

import sys
import argparse
import os
from waldo.data_manipulation import *
import xml.dom.minidom as minidom
import unicodedata

parser = argparse.ArgumentParser(description="Creates line images from page image",
                                 epilog="E.g.  " + sys.argv[0] + "  data/LDC2012T15"
                                 " data/LDC2013T09 data/LDC2013T15 data/madcat.train.raw.lineid "
                                 " data/local/lines ",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('database_path1', type=str,
                    help='Path to the downloaded madcat data directory 1')
parser.add_argument('database_path2', type=str,
                    help='Path to the downloaded madcat data directory 2')
parser.add_argument('database_path3', type=str,
                    help='Path to the downloaded madcat data directory 3')
parser.add_argument('data_splits', type=str,
                    help='Path to file that contains the train/test/dev split information')
parser.add_argument('out_dir', type=str,
                    help='directory location to write output files')
parser.add_argument('writing_condition1', type=str,
                    help='Path to the downloaded (and extracted) writing conditions file 1')
parser.add_argument('writing_condition2', type=str,
                    help='Path to the downloaded (and extracted) writing conditions file 2')
parser.add_argument('writing_condition3', type=str,
                    help='Path to the downloaded (and extracted) writing conditions file 3')
args = parser.parse_args()


def check_file_location(base_name, wc_dict1, wc_dict2, wc_dict3):
    """ Returns the complete path of the page image and corresponding
        xml file.
    Returns
    -------
    image_file_name (string): complete path and name of the page image.
    madcat_file_path (string): complete path and name of the madcat xml file
                               corresponding to the page image.
    """
    madcat_file_path1 = os.path.join(args.database_path1, 'madcat', base_name + '.madcat.xml')
    madcat_file_path2 = os.path.join(args.database_path2, 'madcat', base_name + '.madcat.xml')
    madcat_file_path3 = os.path.join(args.database_path3, 'madcat', base_name + '.madcat.xml')

    image_file_path1 = os.path.join(args.database_path1, 'images', base_name + '.tif')
    image_file_path2 = os.path.join(args.database_path2, 'images', base_name + '.tif')
    image_file_path3 = os.path.join(args.database_path3, 'images', base_name + '.tif')

    if os.path.exists(madcat_file_path1):
        return madcat_file_path1, image_file_path1, wc_dict1

    if os.path.exists(madcat_file_path2):
        return madcat_file_path2, image_file_path2, wc_dict2

    if os.path.exists(madcat_file_path3):
        return madcat_file_path3, image_file_path3, wc_dict3

    return None, None, None


def parse_writing_conditions(writing_conditions):
    """ Given writing condition file path, returns a dictionary which have writing condition
        of each page image.
    Returns
    ------
    (dict): dictionary with key as page image name and value as writing condition.
    """
    with open(writing_conditions) as f:
        file_writing_cond = dict()
        for line in f:
            line_list = line.strip().split("\t")
            file_writing_cond[line_list[0]] = line_list[3]
    return file_writing_cond


def check_writing_condition(wc_dict, base_name):
    """ Given writing condition dictionary, checks if a page image is writing
        in a specified writing condition.
        It is used to create subset of dataset based on writing condition.
    Returns
    (bool): True if writing condition matches.
    """
    #return True
    writing_condition = wc_dict[base_name].strip()
    if writing_condition != 'IUC':
        return False

    return True


def get_file_list():
    """ Given writing condition and data splits, it returns file list
    that will be processed. writing conditions helps in creating subset
    of dataset from full dataset.
    Returns
    []: list of files to be processed.
    """

    wc_dict1 = parse_writing_conditions(args.writing_condition1)
    wc_dict2 = parse_writing_conditions(args.writing_condition2)
    wc_dict3 = parse_writing_conditions(args.writing_condition3)
    splits_handle = open(args.data_splits, 'r')
    splits_data = splits_handle.read().strip().split('\n')

    file_list = list()
    prev_base_name = ''
    for line in splits_data:
        base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
        if prev_base_name != base_name:
            prev_base_name = base_name
            madcat_file_path, image_file_path, wc_dict = check_file_location(base_name, wc_dict1, wc_dict2, wc_dict3)
            if wc_dict is None or not check_writing_condition(wc_dict, base_name):
                continue
            file_list.append(madcat_file_path)

    return file_list


def get_line_mar_from_word_bb(madcat_file_path, mar_text_fh):
    """ Given a page image, extracts the line images from it.
    Input
    -----
    image_file_name (string): complete path and name of the page image.
    madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.
    """

    text_line_word_dict = read_text(madcat_file_path)
    doc = minidom.parse(madcat_file_path)
    zone = doc.getElementsByTagName('zone')
    base_name = os.path.basename(madcat_file_path).split('.madcat')[0]
    for node in zone:
        line_id = node.getAttribute('id')
        line_id = line_id.zfill(4)
        token_image = node.getElementsByTagName('token-image')
        mbb_input = []
        for token_node in token_image:
            word_point = token_node.getElementsByTagName('point')
            for word_node in word_point:
                word_coordinate = (int(word_node.getAttribute('x')), int(word_node.getAttribute('y')))
                mbb_input.append(word_coordinate)
        points = get_minimum_bounding_box(mbb_input)
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points[0], points[1], points[2], points[3]
        min_x, min_y = int(min(x1, x2, x3, x4)), int(min(y1, y2, y3, y4))
        max_x, max_y = int(max(x1, x2, x3, x4)), int(max(y1, y2, y3, y4))

        line = text_line_word_dict[line_id]
        text = ' '.join(line)
        utt_id_filename = base_name + '_' + str(line_id).zfill(4)
        utt_id_coordinates = str(min_x) + '_' + str(min_y) + '_' + str(max_x) + '_' + str(max_y)
        point_str = str()
        for point in points:
            point_str = point_str + str(int(point[0])) + ',' + str(int(point[1])) + ','
        point_str = point_str[:-1]
        mar_text_fh.write(utt_id_filename + ' ' + utt_id_coordinates + ' ' + point_str + ' ' + text + '\n')


def read_text(madcat_file_path):
    """ Maps every word in the page image to a  corresponding line.
    Args:
        madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.
    Returns:
        dict: Mapping every word in the page image to a  corresponding line.
    """

    word_line_dict = dict()
    doc = minidom.parse(madcat_file_path)
    segment = doc.getElementsByTagName('segment')
    zone = doc.getElementsByTagName('zone')
    for node in zone:
        line_id = node.getAttribute('id')
        word_image = node.getElementsByTagName('token-image')
        for tnode in word_image:
            word_id = tnode.getAttribute('id')
            word_line_dict[word_id] = line_id

    text_line_word_dict = dict()
    for node in segment:
        token = node.getElementsByTagName('token')
        for tnode in token:
            ref_word_id = tnode.getAttribute('ref_id')
            word = tnode.getElementsByTagName('source')[0].firstChild.nodeValue
            word = unicodedata.normalize('NFKC',word)
            ref_line_id = word_line_dict[ref_word_id]
            ref_line_id = ref_line_id.zfill(4)
            if ref_line_id not in text_line_word_dict:
                text_line_word_dict[ref_line_id] = list()
            text_line_word_dict[ref_line_id].append(word)
    return text_line_word_dict


def main():
    file_list = get_file_list()
    mar_text_file = os.path.join(args.out_dir, 'mar_text.txt')
    mar_text_fh = open(mar_text_file, 'w', encoding='utf-8')
    for madcat_file_path in file_list:
        get_line_mar_from_word_bb(madcat_file_path, mar_text_fh)


if __name__ == '__main__':
    main()

