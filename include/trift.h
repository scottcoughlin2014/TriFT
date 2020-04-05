#ifndef TRIFT_H
#define TRIFT_H

#include <cmath>
#include <vector>
#include <complex>
#include "pymangle.h"
#include "vector.h"

const double pi = 3.14159265;

void trift(double *x, double *y, double *flux, double *u, double *v,
        double *vis_real, double *vis_imag, int nx, int nu,
        double dx, double dy);

void trift_extended(double *x, double *y, double *flux, double *u, double *v,
        double *vis_real, double *vis_imag, int nx, int nu,
        double dx, double dy);

void trift2D(double *x, double *y, double *flux, double *u, double *v,
        double *vis_real, double *vis_imag, int nx, int nu, int nv,
        double dx, double dy);

#endif
