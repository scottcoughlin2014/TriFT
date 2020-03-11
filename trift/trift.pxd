import cython

cdef extern from "trift.h":
    void trift(double *x, double *y, double *flux, double *u, double *v, 
            complex *vis, int nx, int nuv)
