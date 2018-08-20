#!/usr/bin/env python3

# Copyright 2018 Daniel Povey, Hang Lyu

# Apache 2.0
"""
c_segmenter.py

the C funtion: c_run_segmentation post-process the waldo data
"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to C code
cdef extern void c_run_segmentation (float* class_pred, int class_dim,
                                     float* adj_pred, int offset_dim,
                                     int img_width, int img_height,
                                     int num_classes,
                                     int* offset_list,
                                     int* output,
                                     int* object_class,
                                     float same_different_bias,
                                     float object_merge_factor,
                                     float merge_logprob_bias)

@cython.boundscheck(True)
@cython.wraparound(True)
def run_segmentation(np.ndarray[float, ndim=3, mode="c"] class_pred not None,
                     np.ndarray[float, ndim=3, mode="c"] adj_pred not None,
                     int num_classes,
                     list offset_list not None,
                     float same_different_bias,
                     float object_merge_factor,
                     float merge_logprob_bias) :
    
    """
    For ObjectSegmenter, it needs six arguments
    a. class_pred[]  is a psuedo 4-d numpy array, because the first dim is
                     always 0.
    b. adj_pred[]    which can also be called sameness[]. It is a psuedo 4-d
                     numpy array too.
    c. num_classes   is a int value. It presents the output dimension of nnet.
    d. offset_list   is a list of tuple.
    e. mask_pred     is a (img_height, img_width) matrix. It is used to store
                     result.
    f. object_class  is a (1, num_obj_class), It is used to record the object.
                     class.
    """

    # clip
    epsilon = np.finfo(np.float32).eps
    class_pred = class_pred.clip(epsilon, 1.0 - epsilon)
    adj_pred = adj_pred.clip(epsilon, 1.0 - epsilon)

    # list to numpy array
    # I didn't find a "static typing" now. I convert the tuple list to numpy
    # array outside. "np.array(offset_list)", it will be a 2-d numpy array.
    cdef np.ndarray offset_array = np.array(offset_list).astype(np.int32)

    # other argument
    cdef int class_dim, offset_dim, img_width, img_height, num_class;
    class_dim = class_pred.shape[0];
    offset_dim = adj_pred.shape[0];
    img_height, img_width = adj_pred.shape[1], adj_pred.shape[2];
    num_class = num_classes;

    # output mask and its class label
    cdef np.ndarray mask_pred = np.zeros((img_height,
                                          img_width)).astype(np.int32)
    cdef np.ndarray object_class_pred = np.zeros((1,
                             img_height * img_width)).astype(np.int32)

    c_run_segmentation(<float*> class_pred.data, class_dim,
                       <float*> adj_pred.data, offset_dim,
                       img_width, img_height,
                       num_class,
                       <int*> offset_array.data,
                       <int*> mask_pred.data,
                       <int*> object_class_pred.data,
                       same_different_bias,
                       object_merge_factor,
                       merge_logprob_bias);
    
    object_class = []
    for i in range(object_class_pred.shape[1] - 1):
        if object_class_pred[0, i] == -1:
            break
        object_class.append(object_class_pred[0, i])

    # output
    return mask_pred, object_class

