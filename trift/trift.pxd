import cython

cdef extern from "trift.h":
    void trift(double *x, double *y, double *flux, double *u, double *v, 
            double *vis_real, double *vis_imag, int nx, int nuv)
