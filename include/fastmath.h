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
    /* Approximation from Maass & Martin 2018. We use the last of their 
     * approximations, which is the most accurate, and only adds a small 
     * amount of time compared with the other versions. Paper included in 
     * documentation. */

    double P0 = -0.7763224930; double P1 = -0.03147133771;
    double p0 = 1.776322448; double p1 = 0.2250803518;
    double q1 = 0.4120981204; double q2 = 0.006571619275;
    double lambda = 0.1; double x2 = x*x; double x4 = x2*x2;
    double root_denom = sqrt(1 + lambda*lambda * x2);
    double root_lambda = 0.31622776601683794;

    double P2 = 2 * root_lambda * root_lambda * root_lambda / sqrt(pi) * q2;
    double p2 = 2 * root_lambda / sqrt(pi) * q2;

    return 0.5 / sqrt(root_denom) * ( (p0 + p1*x2 + p2*x4) * FastSin(x) / 
            (1 + q1*x2 + q2*x4) + x / root_denom * (P0 + P1*x2 + P2*x4) * 
            FastCos(x) / (1 + q1*x2 + q2*x4));
}

