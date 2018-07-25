#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("c_segment",
                             sources=["c_segment.pyx", "segment.cc"],
                             language="c++",
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=["-std=c++11"]
                             )],
)
