from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

import numpy

trift = Extension("trift.trift",sources=["trift/trift.pyx","src/trift.cc"],\
        include_dirs=[numpy.get_include(),"./include",\
        "./delaunator-cpp/include"], language="c++", \
        extra_compile_args=['-std=c++11','-Ofast'],\
        extra_link_args=["-std=c++11",'-Ofast'])

setup(cmdclass = {'build_ext': build_ext}, ext_modules = [trift])
