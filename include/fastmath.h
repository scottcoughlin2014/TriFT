double FastCos(double x) {
    /* Approximation based on the discussion found here: 
     * https://stackoverflow.com/questions/18662261/fastest-
     * implementation-of-sine-cosine-and-square-root-in-c-doesnt-need-to-b 
     * Also see documentation. */
    double tp = 1./(2.*pi);

    x *= tp;
    x -= double(.25) + std::floor(x + double(.25));
    x *= double(16.) * (std::abs(x) - double(.5));
    double ax = std::abs(x);
    x += x * (double(0.19521882)*(ax - double(1.)) + 
            double(0.01915497)*(ax*ax - double(1.)));

    return x;
}

double FastSin(double x) {
    return FastCos(x - pi/2.);
}

double BesselJ0(double x) {
    /* Approximation from Tumakov 2019, using their equations 6 and 8. See
     * documentation for additional details.*/

    if (abs(x) < 8) {
        double z = 0.12548309 * x;
        double y = z*z;
        double p1 = y*(y - 2.4015149);

        y = y + p1;

        double p2 = (p1 + 1.1118167)*(y - 0.25900994) + 0.69601147;
        double p3 = (p1 + 1.8671225)*(y + 4.7195298) + 2.0662144;

        return p2 * p3 - 3.4387229;
    }
    else {
        double ax = abs(x);
        double p = 0.63661977/ax;
        double q = p*p;

        return sqrt(p) * (1. + (0.63021018*q - 0.15421257)*q) * 
                FastCos(ax - 0.78539816 + (0.25232973*q - 0.19634954)*p);
    }
}

double BesselJ1(double x) {
    /* Approximation from Tumakov 2019, using their equations 10 and 12. See
     * documentation for additional details.*/

    if (abs(x) < 8) {
        double z = 0.11994381 * x;
        double y = z*z;
        double p1 = y*(y - 2.4087342);

        y = y + p1;

        double p2 = (p1 + 0.57043493)*(y + 6.0949586) + 9.3528179;
        double p3 = p2 * (y + 0.24958757) - 0.18457525;

        return z * (p3 * (p1 + 1.3196524) + 0.18651755);
    }
    else {
        double ax = abs(x);
        double p = 0.63661977 / ax;
        double q = p * p;

        return sqrt(p) * ( 1. + (0.46263771 - 1.1771851*q)*q) * 
                FastCos(ax - 2.3561945 + (0.58904862 - 0.63587091*q)*p);
    }
}

