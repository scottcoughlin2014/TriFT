import matplotlib.tri as tri
import numpy
cimport numpy
cimport cython

################################################################################
# 
# The C++ version.
#
################################################################################

def trift_2D(numpy.ndarray[double, ndim=1, mode="c"] x, \
        numpy.ndarray[double, ndim=1, mode="c"] y, \
        numpy.ndarray[double, ndim=2, mode="c"] flux, \
        numpy.ndarray[double, ndim=1, mode="c"] u, \
        numpy.ndarray[double, ndim=1, mode="c"] v, \
        double dx, double dy, int nthreads=1):

    cdef int nv = flux.shape[1]

    cdef numpy.ndarray[double, ndim=2] vis_real = numpy.zeros((u.size,nv), \
            dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=2] vis_imag = numpy.zeros((u.size,nv), \
            dtype=numpy.double)

    trift2D(&x[0], &y[0], &flux[0,0], &u[0], &v[0], &vis_real[0,0],\
            &vis_imag[0,0], x.size, u.size, nv, dx, dy, nthreads)

    cdef numpy.ndarray[complex, ndim=2] vis = vis_real + 1j*vis_imag

    return vis
