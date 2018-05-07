#!/usr/bin/env python3

# Copyright   2018 Johns Hopkins University (author: Desh Raj)
# Apache 2.0

""" This module will be used for loading ICDAR 2015 data as a class.
 Further functions may be added as and when required.
"""

import os
import numpy as np
from collections import namedtuple
from math import hypot
from PIL import Image,ImageDraw
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
	image_tuple is a named tuple which contains:
				image (np.arr): actual image in numpy array format
				mask (np.arr): numpy array of the same size as the image, 
				mapping each pixel to the object-id.
				id_class (np.arr): numpy array indexed by object-id, returning the
				object-class for that object-id
	"""

	image_tuple = namedtuple('image_tuple', 'image '
								'mask '
								'id_class'
							)

	"""
	data_tuple is a named tuple which contains:
				train_data (list(image_tuple)): training set
				test_data (list(image_tuple)): test set
	"""

	data_tuple = namedtuple('data_tuple', 'train_data '
								'test_data'
							)


	"""
	bounding_box_tuple is a named tuple which contains:
				corner_points (list((int,int))): list of 4 points which make up the box
				text (string): text contained in the bounding box
	"""
	bounding_box_tuple = namedtuple('bounding_box_tuple', 'corner_points '
										'text'
									)


	def __init__(self, data_dir=None):
		"""Constructor for the class.
		Validates the path to ICDAR 2015 data.
		"""
		self.data_dir=""
		if data_dir is None:
			data_dir = self.DATA_DIR

		if not self._validate_path(data_dir):
			raise ValueError("The path is invalid. Either of the following"
				" could be wrong:\n"
				"- Path does not exist.\n"
				"- Path does not point to a directory.\n"
				"- Directory is empty.\n"
				"- The training or test directories have unequal number"
				"of images and labels leading to a mismatch.")
		else:
			self.data_dir = data_dir


	def load_data(self):
		"""Loads the ICDAR data and returns a tuple containing two structures: train and test
		Returns
		--------
		data (image_tuple,image_tuple): train and test image tuples
		"""
		train_data = self._load_data_wrapper(self.tr_img_dir, self.tr_lbl_dir)
		test_data = self._load_data_wrapper(self.te_img_dir, self.te_lbl_dir)

		return self.data_tuple(
			train_data=train_data,
			test_data=test_data
		)


	def _check_images_and_labels(self, image_dir, label_dir):
		"""Checks if the number of images is equal to
		the number of labels in the path.
		Input
		------
		image_dir (string): path to image directory
		label_dir (string): path to label directory
		"""
		return len(os.listdir(image_dir))==len(os.listdir(label_dir))


	def _validate_path(self, data_dir):
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

			if (self._check_images_and_labels(self.tr_img_dir, self.tr_lbl_dir) 
				and self._check_images_and_labels(self.te_img_dir, self.te_lbl_dir)):
				
				return True
		
		return False



	def _load_data_wrapper(self,img_dir,lbl_dir):
		"""Given the image and label directories, returns a structured list.
		Input
		------
		img_dir (string): path to image directory
		lbl_dir (string): path to label directory

		Returns
		-------
		data (list(image_tuple)): list containing image data
		"""
		data = []

		for img,lbl in zip(glob(img_dir+"/*.jpg"),glob(lbl_dir+"/*.txt")):
			im = Image.open(img)
			im_arr = np.array(im)
			lbl_fh = open(lbl,encoding='utf-8')

			bounding_box_list = self._get_bounding_boxes(lbl_fh)
			sorted_bounding_box_list = self._sort_b_b_list_for_mask(bounding_box_list)
			id_class_arr = self._get_object_classes(sorted_bounding_box_list)
			im_mask_arr = self._get_mask_arr(im_arr,sorted_bounding_box_list)
			

			data.append(
				self.image_tuple(
					image=im_arr,
					mask=im_mask_arr,
					id_class=id_class_arr
				)
			)

		return data


	def _get_bounding_boxes(self,label_fh):
		""" Given the file handle of the file containing image data, it returns
		a list of the bounding boxes contained in the image.
		Returns
		--------
		bounding_box_list (list(bounding_box_tuple))
		"""
		bounding_box_list = []
		for line in label_fh.readlines():
			try:
				line = line.replace(u'\ufeff', '')
				if line != '':
					x1, y1, x2, y2, x3, y3, x4, y4= [int(i) for i in line.split(',')[:-1]]
					text = line.split(',')[-1]
					p1 = (x1, y1)
					p2 = (x2, y2)
					p3 = (x3, y3)
					p4 = (x4, y4)
					bounding_box_list.append(
						self.bounding_box_tuple(
							corner_points=[p1,p2,p3,p4],
							text=text
						)
					)
			except:
				pass
		return bounding_box_list


	def _get_object_classes(self,bounding_box_list):
		"""Given the list of bounding boxes, it returns an array mapping object ids to their
		respective classes. Background has class 0 and text has class 1.
		Input
		------
		bounding_box_list (list(bounding_box_tuple))

		Returns
		--------
		id_class_arr (np.arr): numpy array indexed by object id.
		Note: index 0 refers to the background object_id and object_class
		"""
		class_names = np.array([1 for bb in bounding_box_list])
		id_class_arr = np.insert(class_names, 0, 0)
		return id_class_arr


	
	def _sort_b_b_list_for_mask(self,bounding_box_list):
		"""Given a list of bounding boxes, returns a new list sorted in descending order by
		the breadth (shorter side) of the rectangles.
		"""
		sorted_bounding_box_list = sorted(bounding_box_list,
			key=lambda bb: self._get_shorter_side(bb), reverse=True)
		return sorted_bounding_box_list


	def _get_shorter_side(self,bounding_box):
		"""Given a bounding box, returns the length of the shorter side.
		Input
		------
		bounding_box (bounding_box_tuple)

		Returns
		-------
		shorter_side_length (float)
		"""
		return min(
			self._Euclidean_distance(bounding_box.corner_points[0],bounding_box.corner_points[1]),
			self._Euclidean_distance(bounding_box.corner_points[1],bounding_box.corner_points[2])
		)


	def _Euclidean_distance(self,a,b):
		"""Given two points, returns their Euclidean distance.
		Input
		-----
		a ((int,int)): first point
		b ((int,int)): second point
		"""
		return hypot(a[0]-b[0],a[1]-b[1])


	def _get_mask_arr(self,im_arr,bounding_box_list):
		"""Given the image array and the list of bounding boxes, it returns the mask array.
		Input
		------
		im_array (np.arr): original image array
		bounding_box_list (list(bounding_box_tuple)): list of bounding boxes in the image

		Returns
		--------
		im_mask_arr (np.arr): array having same dimensions as im_arr, mapping each pixel 
		to the object-id
		"""
		(height, width, _) = np.shape(im_arr)
		im_mask_arr = np.zeros((height,width))

		for i,bounding_box in enumerate(bounding_box_list):
			im_mask_arr = self._update_mask_arr(im_mask_arr,bounding_box,i+1)

		return im_mask_arr


	def _update_mask_arr(self,im_mask_arr,bounding_box,object_id):
		"""Given a bounding box and its object_id, updates the mask array to set the value of
		all pixels contained in the bounding box to the object_id.
		Input
		------
		im_mask_arr (np.arr): numpy mask array
		bounding_box (bounding_box_tuple): a bounding box
		object_id (int): object_id of the bounding box
		"""
		(height, width) = np.shape(im_mask_arr)
		img = Image.new('L', (width, height), 0)
		ImageDraw.Draw(img).polygon(bounding_box.corner_points, outline=1, fill=1)
		mask_update = np.array(img)
		updated_mask_arr = np.where(mask_update == 1, im_mask_arr, object_id)
		return updated_mask_arr


icdar = ICDAR('/home/desh/Research/icdar/icdar_2015/sample/')
data = icdar.load_data()
im_arr = data.train_data[1].mask
img = Image.fromarray(im_arr,'P')
img.save('image.png')