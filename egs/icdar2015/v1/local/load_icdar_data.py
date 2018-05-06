#!/usr/bin/env python3

# Copyright   2018 Ashish Arora
# Apache 2.0
# minimum bounding box part in this script is originally from
# https://github.com/BebeSparkelSparkel/MinimumBoundingBox

""" This module will be used for loading ICDAR 2015 data as a class.
 Further functions may be added as and when required.

 Author: Desh Raj
"""

import os
import numpy as np
from collections import namedtuple
from math import atan2, cos, sin, pi, degrees, sqrt
from scipy.spatial import ConvexHull
from PIL import Image
from glob import glob


class ICDAR:
	"""Class to load and process the ICDAR 2015 data.
	Initialize with path to directory containing downloaded dataset.
	If no path is provided, the default path /export/b18/draj/icdar_2015/
	will be used.
	"""

	DATA_DIR= "/export/b18/draj/icdar_2015/"
	TRAIN_IMAGES = "ch4_training_images"
	TEST_IMAGES = "ch4_test_images"
	TRAIN_LABELS = "ch4_training_labels"
	TEST_LABELS = "ch4_test_labels"

	"""
	bounding_box is a named tuple which contains:
				 area (float): area of the rectangle
				 length_parallel (float): length of the side that is parallel to unit_vector
				 length_orthogonal (float): length of the side that is orthogonal to unit_vector
				 rectangle_center(int, int): coordinates of the rectangle center
				 (use rectangle_corners to get the corner points of the rectangle)
				 unit_vector (float, float): direction of the length_parallel side.
				 (it's orthogonal vector can be found with the orthogonal_vector function
				 unit_vector_angle (float): angle of the unit vector to be in radians.
				 corner_points [(float, float)]: set that contains the corners of the rectangle
				 text (string): text contained in the bounding box
				 bounding_box_id (string): a unique identifier for the bounding box
	"""

	bounding_box_tuple = namedtuple('bounding_box_tuple', 'area '
										'length_parallel ' 
										'length_orthogonal '
										'rectangle_center ' 
										'unit_vector '
										'unit_vector_angle ' 
										'corner_points '
										'text ' 
										'bounding_box_id'
						 )


	"""
	image_tuple is a named tuple which contains:
				image_filename (string): name of the image file
				image (Image): actual image in Pillow image format
				bounding_box_list (list(bounding_box_tuple)): list of bounding boxes present
				in the image
	"""

	image_tuple = namedtuple('image_tuple', 'image_filename '
								'image '
								'bounding_box_list'
							)

	"""
	data_tuple is a named tuple which contains:
				train_data (list(image_tuple)): training set
				test_data (list(image_tuple)): test set
	"""

	data_tuple = namedtuple('data_tuple', 'train_data '
								'test_data'
							)


	def __init__(self, data_dir=None, padding=400):
		"""Constructor for the class.
		Validates the path to ICDAR 2015 data.
		"""
		self.data_dir=""
		if data_dir is None:
			data_dir = self.DATA_DIR

		if not self.validate_path(data_dir):
			raise ValueError("The path is invalid. Either of the following"
				" could be wrong:\n"
				"- Path does not exist.\n"
				"- Path does not point to a directory.\n"
				"- Directory is empty.\n"
				"- The training or test directories have unequal number"
				"of images and labels leading to a mismatch.")
		else:
			self.data_dir = data_dir
			self.padding = padding
			self.offset = int(self.padding//2)


	def check_images_and_labels(self, image_dir, label_dir):
		"""Checks if the number of images is equal to
		the number of labels in the path.
		Input
		------
		image_dir (string): path to image directory
		label_dir (string): path to label directory
		"""
		return len(os.listdir(image_dir))==len(os.listdir(label_dir))


	def validate_path(self, data_dir):
		"""Checks path for validity.
		Returns
		--------
		is_valid (boolean): True, if path is valid
		"""
		if (os.path.exists(data_dir) 
			and os.path.isdir(data_dir)
			and os.listdir(data_dir)):

			self.tr_img_dir = data_dir + self.TRAIN_IMAGES
			self.tr_lbl_dir = data_dir + self.TRAIN_LABELS
			self.te_img_dir = data_dir + self.TEST_IMAGES
			self.te_lbl_dir = data_dir + self.TEST_LABELS

			if (self.check_images_and_labels(self.tr_img_dir, self.tr_lbl_dir) 
				and self.check_images_and_labels(self.te_img_dir, self.te_lbl_dir)):
				
				return True
		
		return False


	def load_data(self):
		"""Loads the ICDAR data and returns a tuple containing two structures: train and test
		Returns
		--------
		data (image_tuple,image_tuple): train and test image tuples
		"""
		train_data = self.load_data_wrapper(self.tr_img_dir, self.tr_lbl_dir)
		test_data = self.load_data_wrapper(self.te_img_dir, self.te_lbl_dir)

		return self.data_tuple(
			train_data=train_data,
			test_data=test_data
		)


	def load_data_wrapper(self,img_dir, lbl_dir):
		"""Given the image and label directories, returns a structured list.
		Input
		------
		img_dir (string): path to image directory
		lbl_dir (string): path to label directory

		Returns
		-------
		data (list(image_tuple)): list containing images and corresponding data
		"""
		data = []

		for img,lbl in zip(glob(img_dir+"/*.jpg"),glob(lbl_dir+"/*.txt")):
			im_wo_pad = Image.open(img)
			lbl_fh = open(lbl,encoding='utf-8')
			im = self.pad_image(im_wo_pad)

			bounding_box_list = []
			
			for id,line in enumerate(lbl_fh.readlines()):
				try:
					line = line.replace(u'\ufeff', '')
					minimum_bounding_box_input, text = self.get_line_corners(line.strip()) 
					updated_mbb_input = self.update_minimum_bounding_box_input(minimum_bounding_box_input)
					bounding_box = self.minimum_bounding_box(updated_mbb_input)

					p1, p2, p3, p4 = bounding_box.corner_points
					x1, y1 = p1
					x2, y2 = p2
					x3, y3 = p3
					x4, y4 = p4
					min_x = int(min(x1, x2, x3, x4))
					min_y = int(min(y1, y2, y3, y4))
					max_x = int(max(x1, x2, x3, x4))
					max_y = int(max(y1, y2, y3, y4))
					box = (min_x, min_y, max_x, max_y)
					region_initial = im.crop(box)
					rot_points = []
					p1_new = (x1 - min_x, y1 - min_y)
					p2_new = (x2 - min_x, y2 - min_y)
					p3_new = (x3 - min_x, y3 - min_y)
					p4_new = (x4 - min_x, y4 - min_y)
					rot_points.append(p1_new)
					rot_points.append(p2_new)
					rot_points.append(p3_new)
					rot_points.append(p4_new)

					base_name = os.path.splitext(os.path.basename(img))[0]
					line_id = '_' + str(id).zfill(4)

					cropped_bounding_box = self.bounding_box_tuple(bounding_box.area,
							bounding_box.length_parallel,
							bounding_box.length_orthogonal,
							bounding_box.length_orthogonal,
							bounding_box.unit_vector,
							bounding_box.unit_vector_angle,
							set(rot_points),
							text,
							base_name+line_id
						)

					# bounding_box_list.append(cropped_bounding_box)


				except:
					print("Error in file",img)
					break

			tmp = self.image_tuple(
				image_filename=img,
				image=im,
				bounding_box_list=bounding_box_list
			)
			
			# data.append(tmp)

		return data




	def unit_vector(self,pt0, pt1):
		""" Given two points pt0 and pt1, return a unit vector that
			points in the direction of pt0 to pt1.
		Returns
		-------
		(float, float): unit vector
		"""
		dis_0_to_1 = sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
		return (pt1[0] - pt0[0]) / dis_0_to_1, \
			   (pt1[1] - pt0[1]) / dis_0_to_1


	def orthogonal_vector(self,vector):
		""" Given a vector, returns a orthogonal/perpendicular vector of equal length.
		Returns
		------
		(float, float): A vector that points in the direction orthogonal to vector.
		"""
		return -1 * vector[1], vector[0]


	def bounding_area(self,index, hull):
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
		unit_vector_p = self.unit_vector(hull[index], hull[index+1])
		unit_vector_o = self.orthogonal_vector(unit_vector_p)

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


	def to_xy_coordinates(self,unit_vector_angle, point):
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


	def rotate_points(self,center_of_rotation, angle, points):
		""" Rotates a point cloud around the center_of_rotation point by angle
		input
		-----
		center_of_rotation (float, float): angle of unit vector to be in radians.
		angle (float): angle of rotation to be in radians.
		points [(float, float)]: Points to be a list or tuple of points. Points to be rotated.
		Returns
		------
		[(float, float)]: Rotated points around center of rotation by angle
		"""
		rot_points = []
		ang = []
		for pt in points:
			diff = tuple([pt[d] - center_of_rotation[d] for d in range(2)])
			diff_angle = atan2(diff[1], diff[0]) + angle
			ang.append(diff_angle)
			diff_length = sqrt(sum([d**2 for d in diff]))
			rot_points.append((center_of_rotation[0] + diff_length * cos(diff_angle),
							   center_of_rotation[1] + diff_length * sin(diff_angle)))

		return rot_points


	def rectangle_corners(self,rectangle):
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

		return self.rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)


	# use this function to find the listed properties of the minimum bounding box of a point cloud
	def minimum_bounding_box(self,points):
		""" Given a list of 2D points, it returns the minimum area rectangle bounding all
			the points in the point cloud.
		Returns
		------
		returns a namedtuple that contains:
		area: area of the rectangle
		length_parallel: length of the side that is parallel to unit_vector
		length_orthogonal: length of the side that is orthogonal to unit_vector
		rectangle_center: coordinates of the rectangle center
		unit_vector: direction of the length_parallel side. RADIANS
		unit_vector_angle: angle of the unit vector
		corner_points: set that contains the corners of the rectangle
		"""

		if len(points) <= 2: raise ValueError('More than two points required.')

		hull_ordered = [points[index] for index in ConvexHull(points).vertices]
		hull_ordered.append(hull_ordered[0])
		hull_ordered = tuple(hull_ordered)

		min_rectangle = self.bounding_area(0, hull_ordered)
		for i in range(1, len(hull_ordered)-1):
			rectangle = self.bounding_area(i, hull_ordered)
			if rectangle['area'] < min_rectangle['area']:
				min_rectangle = rectangle

		min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
		min_rectangle['rectangle_center'] = self.to_xy_coordinates(min_rectangle['unit_vector_angle'], min_rectangle['rectangle_center'])

		return self.bounding_box_tuple(
			area=min_rectangle['area'],
			length_parallel=min_rectangle['length_parallel'],
			length_orthogonal=min_rectangle['length_orthogonal'],
			rectangle_center=min_rectangle['rectangle_center'],
			unit_vector=min_rectangle['unit_vector'],
			unit_vector_angle=min_rectangle['unit_vector_angle'],
			corner_points=set(self.rectangle_corners(min_rectangle)),
			text='',
			bounding_box_id=''
		)


	def get_center(self,im):
		""" Given image, returns the location of center pixel
		Returns
		-------
		(int, int): center of the image
		"""
		center_x = im.size[0] / 2
		center_y = im.size[1] / 2
		return int(center_x), int(center_y)


	def get_horizontal_angle(self,unit_vector_angle):
		""" Given an angle in radians, returns angle of the unit vector in
			first or fourth quadrant.
		Returns
		------
		(float): updated angle of the unit vector to be in radians.
				 It is only in first or fourth quadrant.
		"""
		if unit_vector_angle > pi / 2 and unit_vector_angle <= pi:
			unit_vector_angle = unit_vector_angle - pi
		elif unit_vector_angle > -pi and unit_vector_angle < -pi / 2:
			unit_vector_angle = unit_vector_angle + pi

		return unit_vector_angle


	def get_smaller_angle(self,bounding_box):
		""" Given a rectangle, returns its smallest absolute angle from horizontal axis.
		Returns
		------
		(float): smallest angle of the rectangle to be in radians.
		"""
		unit_vector = bounding_box.unit_vector
		unit_vector_angle = bounding_box.unit_vector_angle
		ortho_vector = self.orthogonal_vector(unit_vector)
		ortho_vector_angle = atan2(ortho_vector[1], ortho_vector[0])

		unit_vector_angle_updated = self.get_horizontal_angle(unit_vector_angle)
		ortho_vector_angle_updated = self.get_horizontal_angle(ortho_vector_angle)

		if abs(unit_vector_angle_updated) < abs(ortho_vector_angle_updated):
			return unit_vector_angle_updated
		else:
			return ortho_vector_angle_updated


	def rotated_points(self,bounding_box, center):
		""" Given the rectangle, returns corner points of rotated rectangle.
			It rotates the rectangle around the center by its smallest angle.
		Returns
		------- 
		[(int, int)]: 4 corner points of rectangle.
		"""
		p1, p2, p3, p4 = bounding_box.corner_points
		x1, y1 = p1
		x2, y2 = p2
		x3, y3 = p3
		x4, y4 = p4
		center_x, center_y = center
		rotation_angle_in_rad = -(self.get_smaller_angle(bounding_box))
		x_dash_1 = (x1 - center_x) * cos(rotation_angle_in_rad) - (y1 - center_y) * sin(rotation_angle_in_rad) + center_x
		x_dash_2 = (x2 - center_x) * cos(rotation_angle_in_rad) - (y2 - center_y) * sin(rotation_angle_in_rad) + center_x
		x_dash_3 = (x3 - center_x) * cos(rotation_angle_in_rad) - (y3 - center_y) * sin(rotation_angle_in_rad) + center_x
		x_dash_4 = (x4 - center_x) * cos(rotation_angle_in_rad) - (y4 - center_y) * sin(rotation_angle_in_rad) + center_x

		y_dash_1 = (y1 - center_y) * cos(rotation_angle_in_rad) + (x1 - center_x) * sin(rotation_angle_in_rad) + center_y
		y_dash_2 = (y2 - center_y) * cos(rotation_angle_in_rad) + (x2 - center_x) * sin(rotation_angle_in_rad) + center_y
		y_dash_3 = (y3 - center_y) * cos(rotation_angle_in_rad) + (x3 - center_x) * sin(rotation_angle_in_rad) + center_y
		y_dash_4 = (y4 - center_y) * cos(rotation_angle_in_rad) + (x4 - center_x) * sin(rotation_angle_in_rad) + center_y
		return x_dash_1, y_dash_1, x_dash_2, y_dash_2, x_dash_3, y_dash_3, x_dash_4, y_dash_4


	def pad_image(self,image):
		""" Given an image, returns a padded image around the border.
			This routine save the code from crashing if bounding boxes that are
			slightly outside the page boundary.
		Returns
		-------
		image: page image
		"""
		padded_image = Image.new('RGB', (image.size[0] + self.padding, 
			image.size[1] + self.padding), "white")
		padded_image.paste(im=image, box=(self.offset, self.offset))
		return padded_image


	def update_minimum_bounding_box_input(self,bounding_box_input):
		""" Given list of 2D points, returns list of 2D points shifted by an offset.
		Returns
		------
		points [(float, float)]: points, a list or tuple of 2D coordinates
		"""
		updated_minimum_bounding_box_input = []
		for point in bounding_box_input:
			x, y = point
			new_x = x + self.offset
			new_y = y + self.offset
			word_coordinate = (new_x, new_y)
			updated_minimum_bounding_box_input.append(word_coordinate)

		return updated_minimum_bounding_box_input


	def get_line_corners(self,line):
		""" Given a line label, extract corner points and text from it.
		Input
		------
		line (string): 1 line from the labels file
		"""
		if line != '':
			x1, y1, x2, y2, x3, y3, x4, y4= [int(i) for i in line.split(',')[:-1]]
			text = line.split(',')[-1]
			p1 = (x1, y1)
			p2 = (x2, y2)
			p3 = (x3, y3)
			p4 = (x4, y4)
			return [p1, p2, p3, p4], text


	# def set_line_image_data(output_directory, image, line_id, image_path, text, image_fh, lbl_fh):
	# 	""" Given an image, saves a flipped line image. Line image file name
	# 		is formed by appending the line id at the end page image name.
	# 	"""

	# 	base_name = os.path.splitext(os.path.basename(image_path))[0]
	# 	line_id = '_' + str(line_id).zfill(4)
	# 	line_image_path = base_name + line_id + '.tif'
	# 	image_path = os.path.join(output_directory, line_image_path)
	# 	imgray = image.convert('L')
	# 	imgray_rev_arr = np.fliplr(imgray)
	# 	imgray_rev = toimage(imgray_rev_arr)
	# 	imgray_rev.save(image_path)
	# 	image_fh.write(image_path + '\n')
	# 	lbl_fh.write(text + '\n')






icdar = ICDAR(data_dir='/home/desh/Research/icdar/icdar_2015/')
data = icdar.load_data()
