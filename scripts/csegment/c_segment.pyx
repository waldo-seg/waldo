#!/usr/bin/env python

"""
c_segment.py

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
                     np.ndarray[int, ndim=2, mode="c"] offset_list not None,
                     np.ndarray[int, ndim=2, mode="c"] output not None,
                     np.ndarray[int, ndim=2, mode="c"] object_class not None,
                     float same_different_bias,
                     float object_merge_factor,
                     float merge_logprob_bias) :
    
    """
    For ObjectSegmenter, it needs six arguments
    a. class_pred[]  is a psuedo 4-d numpy array, because the first dim is
                     always 0.
    b. adj_pred[]  which can also be called sameness[]. It is a psuedo 4-d numpy
                   array too.
    c. num_classes  is a int value. It presents the output dimension of nnet.
    d. offset_list  is a list of tuple. I didn't find a "static typing" now. I 
                    convert the tuple list to numpy array outside.
                    "np.array(offset_list)", it will be a 2-d numpy array.
    e. output is a (img_height, img_width) matrix. It is used to store result.
    f. object_class  is a (1, num_obj_class), It is used to record the object.
                     class.
    """

    cdef int class_dim, offset_dim, img_width, img_height, num_class;
    
    class_dim = class_pred.shape[0];
    offset_dim = adj_pred.shape[0];
    img_height, img_width = adj_pred.shape[1], adj_pred.shape[2];
    num_class = num_classes;

    c_run_segmentation(<float*> class_pred.data, class_dim,
                       <float*> adj_pred.data, offset_dim,
                       img_width, img_height,
                       num_class,
                       <int*> offset_list.data,
                       <int*> output.data,
                       <int*> object_class.data,
                       same_different_bias,
                       object_merge_factor,
                       merge_logprob_bias);

    return None

