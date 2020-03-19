import matplotlib.tri as tri
import numpy
cimport numpy
cimport cython

################################################################################
# 
# The C++ version.
#
################################################################################

def trift_c(numpy.ndarray[double, ndim=1, mode="c"] x, \
        numpy.ndarray[double, ndim=1, mode="c"] y, \
        numpy.ndarray[double, ndim=1, mode="c"] flux, \
        numpy.ndarray[double, ndim=1, mode="c"] u, \
        numpy.ndarray[double, ndim=1, mode="c"] v, \
        double dx, double dy):

    cdef numpy.ndarray[double, ndim=1] vis_real = numpy.zeros((u.size,), \
            dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=1] vis_imag = numpy.zeros((u.size,), \
            dtype=numpy.double)

    trift(&x[0], &y[0], &flux[0], &u[0], &v[0], &vis_real[0], &vis_imag[0], \
            x.size, u.size, 1, dx, dy)

    cdef numpy.ndarray[complex, ndim=1] vis = vis_real + 1j*vis_imag

    return vis

def trift_2D(numpy.ndarray[double, ndim=1, mode="c"] x, \
        numpy.ndarray[double, ndim=1, mode="c"] y, \
        numpy.ndarray[double, ndim=2, mode="c"] flux, \
        numpy.ndarray[double, ndim=1, mode="c"] u, \
        numpy.ndarray[double, ndim=1, mode="c"] v, \
        double dx, double dy):

    cdef int nv = flux.shape[1]

    cdef numpy.ndarray[double, ndim=2] vis_real = numpy.zeros((u.size,nv), \
            dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=2] vis_imag = numpy.zeros((u.size,nv), \
            dtype=numpy.double)

    trift(&x[0], &y[0], &flux[0,0], &u[0], &v[0], &vis_real[0,0], \
            &vis_imag[0,0], x.size, u.size, nv, dx, dy)

    cdef numpy.ndarray[complex, ndim=2] vis = vis_real + 1j*vis_imag

    return vis

################################################################################
# 
# The Python version.
#
################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def trift_python(numpy.ndarray[double, ndim=1] x, \
        numpy.ndarray[double, ndim=1] y, \
        numpy.ndarray[double, ndim=1] flux, numpy.ndarray[double, ndim=1] u, \
        numpy.ndarray[double, ndim=1] v):

    cdef unsigned int i, j, ntriangles

    # Merge x and y and u and v together.

    cdef numpy.ndarray[double, ndim=2] grid = numpy.vstack((x, y)).T
    cdef numpy.ndarray[double, ndim=2] uv = numpy.vstack((u, v))

    # Do a triangulation of the grid.

    triang = tri.Triangulation(x, y)
    cdef numpy.ndarray[int, ndim=2] vertices = triang.triangles
    ntriangles = triang.triangles.shape[0]

    # Now loop through and do the calculation.

    cdef numpy.ndarray[complex, ndim=1] vis = numpy.zeros(u.size, dtype=complex)
    cdef numpy.ndarray[double, ndim=1] r_n1, r_n, r_n_1, ln, ln_1, ln3D, ln_13D
    cdef numpy.ndarray[double, ndim=1] zhat = numpy.array([0.,0.,1.])
    cdef numpy.ndarray[complex, ndim=1] F_triang
    cdef double intensity_triang, cross_dot_ln_13D

    ln3D = numpy.array([0.,0.,0.])
    ln_13D = numpy.array([0.,0.,0.])

    for j in range(ntriangles):
        intensity_triang = 0.
        F_triang = numpy.zeros(u.size, dtype=complex)

        for i in range(3):
            intensity_triang += flux[vertices[j,i]] / 3.

            r_n1 = grid[vertices[j,(i+2)%3]]
            r_n = grid[vertices[j,i]]
            r_n_1 = grid[vertices[j,(i+1)%3]]

            # Calculate the side vectors.

            ln = r_n1 - r_n
            ln_1 = r_n - r_n_1

            ln3D = numpy.array([ln[0],ln[1],0])
            ln_13D = numpy.array([ln_1[0],ln_1[1],0])

            # Now calculate the Fourier transform of the triangle.

            F_triang += numpy.dot(numpy.cross(zhat, ln3D), ln_13D) / \
                    (numpy.dot(ln, uv) * numpy.dot(ln_1, uv))*\
                    numpy.exp(-1j*2*numpy.pi * numpy.dot(r_n, uv))

        vis += F_triang * intensity_triang

    return vis
