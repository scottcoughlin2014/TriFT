import cython

cdef extern from "trift.h":
    void trift(double *x, double *y, double *flux, double *u, double *v, 
            double *vis_real, double *vis_imag, int nx, int nuv,
            double dx, double dy, int nthreads)

    void trift_extended(double *x, double *y, double *flux, double *u, 
            double *v, double *vis_real, double *vis_imag, int nx, int nuv,
            double dx, double dy)

    void trift2D(double *x, double *y, double *flux, double *u, double *v, 
            double *vis_real, double *vis_imag, int nx, int nuv, int nv,
            double dx, double dy, int nthreads)

    void trift2D_extended(double *x, double *y, double *flux, double *u, 
            double *v, double *vis_real, double *vis_imag, int nx, int nuv, 
            int nv, double dx, double dy)
