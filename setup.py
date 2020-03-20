from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

import numpy

os.environ["CC"] = "gcc-9"
os.environ["CXX"] = "gcc-9"

trift = Extension("trift.trift",sources=["trift/trift.pyx","src/trift.cc"],\
        include_dirs=[numpy.get_include(),"./include"], language="c++", \
        extra_compile_args=['-std=c++11','-O3','-fopenmp'],\
        extra_link_args=["-std=c++11",'-O3','-fopenmp'])

setup(cmdclass = {'build_ext': build_ext}, ext_modules = [trift])
