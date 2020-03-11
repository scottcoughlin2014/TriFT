#ifndef TRIFT_H
#define TRIFT_H

#include <cmath>
#include <vector>
#include <complex>
#include "pymangle.h"
#include "vector.h"

const double pi = 3.14159265;

void trift(double *x, double *y, double *flux, double *u, double *v, \
        std::complex<double> *vis, int nx, int nu);

std::complex<double> fast_exp(std::complex<double> x, int n);

#endif
