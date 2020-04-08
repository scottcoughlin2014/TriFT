#include "trift.h"
#include <delaunator.hpp>
#include "timer.c"

void trift(double *x, double *y, double *flux, double *u, double *v,
        double *vis_real, double *vis_imag, int nx, int nu, double dx, 
        double dy) {

    // Set up the coordinates for the triangulation.

    std::vector<double> coords;

    for (int i=0; i < nx; i++) {
        coords.push_back(x[i]);
        coords.push_back(y[i]);
    }

    // Run the Delauney triangulation here.

    delaunator::Delaunator d(coords);

    // Loop through and take the Fourier transform of each triangle.
    
    Vector<double, 3> zhat(0., 0., 1.);

    for (std::size_t i = 0; i < d.triangles.size(); i+=3) {
        double intensity_triangle = (flux[d.triangles[i]] + 
            flux[d.triangles[i+1]] + flux[d.triangles[i+2]]) / 3.;

        for (int j = 0; j < 3; j++) {
            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = d.triangles[i + (j+1)%3];
            Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

            std::size_t i_rn = d.triangles[i + j];
            Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

            std::size_t i_rn_1 = d.triangles[i + (j+2)%3];
            Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 3> ln = rn1 - rn;
            Vector<double, 3> ln_1 = rn - rn_1;

            // Now loop through the UV points and calculate the Fourier
            // Transform.

            double ln_1_dot_zhat_cross_ln = ln_1.dot(zhat.cross(ln));

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                Vector <double, 3> uv(2*pi*u[k], 2*pi*v[k], 0.);

                double rn_dot_uv = rn.dot(uv);
                
                vis_real[k] += intensity_triangle * ln_1_dot_zhat_cross_ln /
                    (ln.dot(uv) * ln_1.dot(uv)) * cos(rn_dot_uv);
                vis_imag[k] += intensity_triangle * ln_1_dot_zhat_cross_ln /
                    (ln.dot(uv) * ln_1.dot(uv)) * sin(rn_dot_uv);
            }
        }
    }

    // Do the centering of the data.

    Vector<double, 2> center(-dx, -dy);

    //TCLEAR(moo); TSTART(moo);
    for (std::size_t i = 0; i < (std::size_t) nu; i++) {
        Vector <double, 2> uv(2*pi*u[i], 2*pi*v[i]);

        double vis_real_temp = vis_real[i]*cos(center.dot(uv)) - vis_imag[i]*
            sin(center.dot(uv));
        double vis_imag_temp = vis_real[i]*sin(center.dot(uv)) + vis_imag[i]*
            cos(center.dot(uv));

        vis_real[i] = vis_real_temp;
        vis_imag[i] = vis_imag_temp;
    }
    //TSTOP(moo);
    //printf("%f\n", TGIVE(moo));

    // Clean up.

    //delete[] rn_dot_uv; delete[] sin_rn_dot_uv; delete[] cos_rn_dot_uv;
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
                cos(ax - 0.78539816 + (0.25232973*q - 0.19634954)*p);
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

    return 0.5 / sqrt(root_denom) * ( (p0 + p1*x2 + p2*x4) * sin(x) / 
            (1 + q1*x2 + q2*x4) + x / root_denom * (P0 + P1*x2 + P2*x4) * 
            cos(x) / (1 + q1*x2 + q2*x4));
}

void trift_extended(double *x, double *y, double *flux, double *u, double *v,
        double *vis_real, double *vis_imag, int nx, int nu, double dx, 
        double dy) {

    // Set up the coordinates for the triangulation.

    std::vector<double> coords;

    for (int i=0; i < nx; i++) {
        coords.push_back(x[i]);
        coords.push_back(y[i]);
    }

    // Run the Delauney triangulation here.

    delaunator::Delaunator d(coords);

    // Loop through and take the Fourier transform of each triangle.
    
    std::complex<double> I = std::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    Vector<std::complex<double>, 3> *integral = new Vector<std::complex<double>,
        3>[nu];

    Vector<double, 3> *bessel1 = new Vector<double, 3>[nu*3];
    Vector<double, 3> *bessel0_prefix1 = new Vector<double, 3>[nu*3];
    Vector<double, 3> *bessel0_prefix2 = new Vector<double, 3>[nu*3];
    double *bessel0_prefix3 = new double[nu*3];
    Vector<double, 3> *bessel0_prefix4 = new Vector<double, 3>[nu*3];
    double *bessel0 = new double[nu*3];
    std::complex<double> *exp_part = new std::complex<double>[nu*3];

    //TCREATE(moo); TCLEAR(moo);
    for (std::size_t i = 0; i < d.triangles.size(); i+=3) {
        // Calculate the area of the triangle.

        Vector<double, 3> Vertex1(x[d.triangles[i+0]], y[d.triangles[i+0]], 0.);
        Vector<double, 3> Vertex2(x[d.triangles[i+1]], y[d.triangles[i+1]], 0.);
        Vector<double, 3> Vertex3(x[d.triangles[i+2]], y[d.triangles[i+2]], 0.);

        Vector<double, 3> Side1 = Vertex2 - Vertex1;
        Vector<double, 3> Side2 = Vertex3 - Vertex1;

        double Area = 0.5 * (Side1.cross(Side2)).norm();

        // Precompute some aspects of the integral that remain the same.

        //TSTART(moo);
        for (int m = 0; m < 3; m++) {
            // Get the appropriate vertices.

            std::size_t i_rm1 = d.triangles[i + (m+1)%3];
            Vector<double, 3> rm1(x[i_rm1], y[i_rm1],  0.);

            std::size_t i_rm = d.triangles[i + m];
            Vector<double, 3> rm(x[i_rm], y[i_rm],  0.);

            // Calculate the needed derivatives of those.

            Vector<double, 3> lm = rm1 - rm;
            Vector<double, 3> r_mc = 0.5 * (rm1 + rm);
            Vector<double, 3> zhat_cross_lm = zhat.cross(lm);

            // Now loop through the uv points and calculate the pieces of the 
            // integral.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                Vector <double, 3> uv(2*pi*u[k],2*pi*v[k],0.);

                double zhat_dot_lm_cross_uv = zhat.dot(lm.cross(uv));

                bessel0_prefix1[m*nu+k] = zhat_cross_lm;
                bessel0_prefix2[m*nu+k] = r_mc * zhat_dot_lm_cross_uv;
                bessel0_prefix3[m*nu+k] = zhat_dot_lm_cross_uv;
                bessel0_prefix4[m*nu+k] = 2.*uv/uv.dot(uv)*zhat_dot_lm_cross_uv;

                bessel0[m*nu + k] = BesselJ0(uv.dot(lm)/2.);

                bessel1[m*nu + k] = lm * (zhat.dot(lm.cross(uv))/2.) * 
                        BesselJ1(uv.dot(lm)/2.);

                exp_part[m*nu + k] = (cos(r_mc.dot(uv))+ I*sin(r_mc.dot(uv))) / 
                        (uv.dot(uv));
            }
        }
        //TSTOP(moo);

        // Now loop through an do the actual calculation.

        for (int j = 0; j < 3; j++) {
            double intensity = flux[d.triangles[i + j]];

            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = d.triangles[i + (j+1)%3];
            Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

            std::size_t i_rn = d.triangles[i + j];
            Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

            std::size_t i_rn_1 = d.triangles[i + (j+2)%3];
            Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 3> ln1 = rn_1 - rn1;

            // Now loop through the UV points and calculate the Fourier
            // Transform.

            Vector<double, 3> zhat_cross_ln1 = zhat.cross(ln1);

            for (int m = 0; m < 3; m++) {
                for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                    integral[k] += ((bessel0_prefix1[m*nu+k] + 
                            I*bessel0_prefix2[m*nu+k] - 
                            I*bessel0_prefix3[m*nu+k]*rn1 - 
                            bessel0_prefix4[m*nu+k]) * bessel0[m*nu+k] - 
                            bessel1[m*nu+k]) * exp_part[m*nu+k];
                }
            }

            // Finally put into the real and imaginary components, and clean
            // out the integral array.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                vis_real[k] += (intensity * zhat_cross_ln1.dot(integral[k]) / 
                    (2.*Area)).real();
                vis_imag[k] += (intensity * zhat_cross_ln1.dot(integral[k]) / 
                    (2.*Area)).imag();

                integral[k] = 0.;
            }
        }
    }
    //printf("%f\n", TGIVE(moo));

    // Do the centering of the data.

    Vector<double, 2> center(-dx, -dy);

    //TCLEAR(moo); TSTART(moo);
    for (std::size_t i = 0; i < (std::size_t) nu; i++) {
        Vector <double, 2> uv(2*pi*u[i], 2*pi*v[i]);

        double vis_real_temp = vis_real[i]*cos(center.dot(uv)) - vis_imag[i]*
            sin(center.dot(uv));
        double vis_imag_temp = vis_real[i]*sin(center.dot(uv)) + vis_imag[i]*
            cos(center.dot(uv));

        vis_real[i] = vis_real_temp;
        vis_imag[i] = vis_imag_temp;
    }
    //TSTOP(moo);
    //printf("%f\n", TGIVE(moo));

    // Clean up.

    delete[] integral; delete[] bessel1; delete[] bessel0_prefix1;
    delete[] bessel0_prefix2; delete[] bessel0_prefix3; 
    delete[] bessel0_prefix4; delete[] bessel0; delete[] exp_part;

}

void trift2D(double *x, double *y, double *flux, double *u, double *v,
        double *vis_real, double *vis_imag, int nx, int nu, int nv,
        double dx, double dy) {

    // Set up the coordinates for the triangulation.
    
    std::vector<double> coords;

    for (int i=0; i < nx; i++) {
        coords.push_back(x[i]);
        coords.push_back(y[i]);
    }

    // Run the Delauney triangulation here.

    delaunator::Delaunator d(coords);

    // Loop through and take the Fourier transform of each triangle.
    
    Vector<double, 3> zhat(0., 0., 1.);

    double *intensity_triangle = new double[nv];

    for (std::size_t i = 0; i < d.triangles.size(); i+=3) {
        // Get the intensity of the triangle at each wavelength.

        for (std::size_t l = 0; l < (std::size_t) nv; l++) {
            intensity_triangle[l] = (flux[d.triangles[i]*nv+l] + 
                flux[d.triangles[i+1]*nv+l] + flux[d.triangles[i+2]*nv+l]) / 3.;
        }

        // Calculate the FT

        for (int j = 0; j < 3; j++) {
            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = d.triangles[i + (j+1)%3];
            Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

            std::size_t i_rn = d.triangles[i + j];
            Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

            std::size_t i_rn_1 = d.triangles[i + (j+2)%3];
            Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 3> ln = rn1 - rn;
            Vector<double, 3> ln_1 = rn - rn_1;

            // Now loop through the UV points and calculate the Fourier
            // Transform.

            double ln_1_dot_zhat_cross_ln = ln_1.dot(zhat.cross(ln));

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                Vector <double, 3> uv(2*pi*u[k], 2*pi*v[k], 0.);

                double rn_dot_uv = rn.dot(uv);
                
                std::size_t idy = k * nv;

                for (std::size_t l = 0; l < (std::size_t) nv; l++) {
                    vis_real[idy+l] += intensity_triangle[l] * 
                        ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                        cos(rn_dot_uv);
                    vis_imag[idy+l] += intensity_triangle[l] * 
                        ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                        sin(rn_dot_uv);
                }
            }
        }
    }

    // Do the centering of the data.

    Vector<double, 2> center(-dx, -dy);

    //TCLEAR(moo); TSTART(moo);
    for (std::size_t i = 0; i < (std::size_t) nu; i++) {
        Vector <double, 2> uv(2*pi*u[i], 2*pi*v[i]);

        for (std::size_t j = 0; j < (std::size_t) nv; j++) {

            double vis_real_temp = vis_real[i*nv+j]*cos(center.dot(uv)) - 
                vis_imag[i*nv+j]*sin(center.dot(uv));
            double vis_imag_temp = vis_real[i*nv+j]*sin(center.dot(uv)) + 
                vis_imag[i*nv+j]*cos(center.dot(uv));

            vis_real[i*nv+j] = vis_real_temp;
            vis_imag[i*nv+j] = vis_imag_temp;
        }
    }
    //TSTOP(moo);
    //printf("%f\n", TGIVE(moo));

    // Clean up.

    //delete[] rn_dot_uv; delete[] sin_rn_dot_uv; delete[] cos_rn_dot_uv;
    delete[] intensity_triangle;
}

void trift2D_extended(double *x, double *y, double *flux, double *u, double *v,
        double *vis_real, double *vis_imag, int nx, int nu, int nv,
        double dx, double dy) {

    // Set up the coordinates for the triangulation.

    std::vector<double> coords;

    for (int i=0; i < nx; i++) {
        coords.push_back(x[i]);
        coords.push_back(y[i]);
    }

    // Run the Delauney triangulation here.

    delaunator::Delaunator d(coords);

    // Loop through and take the Fourier transform of each triangle.
    
    std::complex<double> I = std::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    Vector<std::complex<double>, 3> *integral = new Vector<std::complex<double>,
        3>[nu];

    Vector<double, 3> *bessel1 = new Vector<double, 3>[nu*3];
    Vector<double, 3> *bessel0_prefix1 = new Vector<double, 3>[nu*3];
    Vector<double, 3> *bessel0_prefix2 = new Vector<double, 3>[nu*3];
    double *bessel0_prefix3 = new double[nu*3];
    Vector<double, 3> *bessel0_prefix4 = new Vector<double, 3>[nu*3];
    double *bessel0 = new double[nu*3];
    std::complex<double> *exp_part = new std::complex<double>[nu*3];

    //TCREATE(moo); TCLEAR(moo);
    for (std::size_t i = 0; i < d.triangles.size(); i+=3) {
        // Calculate the area of the triangle.

        Vector<double, 3> Vertex1(x[d.triangles[i+0]], y[d.triangles[i+0]], 0.);
        Vector<double, 3> Vertex2(x[d.triangles[i+1]], y[d.triangles[i+1]], 0.);
        Vector<double, 3> Vertex3(x[d.triangles[i+2]], y[d.triangles[i+2]], 0.);

        Vector<double, 3> Side1 = Vertex2 - Vertex1;
        Vector<double, 3> Side2 = Vertex3 - Vertex1;

        double Area = 0.5 * (Side1.cross(Side2)).norm();

        // Precompute some aspects of the integral that remain the same.

        //TSTART(moo);
        for (int m = 0; m < 3; m++) {
            // Get the appropriate vertices.

            std::size_t i_rm1 = d.triangles[i + (m+1)%3];
            Vector<double, 3> rm1(x[i_rm1], y[i_rm1],  0.);

            std::size_t i_rm = d.triangles[i + m];
            Vector<double, 3> rm(x[i_rm], y[i_rm],  0.);

            // Calculate the needed derivatives of those.

            Vector<double, 3> lm = rm1 - rm;
            Vector<double, 3> r_mc = 0.5 * (rm1 + rm);
            Vector<double, 3> zhat_cross_lm = zhat.cross(lm);

            // Now loop through the uv points and calculate the pieces of the 
            // integral.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                Vector <double, 3> uv(2*pi*u[k],2*pi*v[k],0.);

                double zhat_dot_lm_cross_uv = zhat.dot(lm.cross(uv));

                bessel0_prefix1[m*nu+k] = zhat_cross_lm;
                bessel0_prefix2[m*nu+k] = r_mc * zhat_dot_lm_cross_uv;
                bessel0_prefix3[m*nu+k] = zhat_dot_lm_cross_uv;
                bessel0_prefix4[m*nu+k] = 2.*uv/uv.dot(uv)*zhat_dot_lm_cross_uv;

                bessel0[m*nu + k] = BesselJ0(uv.dot(lm)/2.);

                bessel1[m*nu + k] = lm * (zhat.dot(lm.cross(uv))/2.) * 
                        BesselJ1(uv.dot(lm)/2.);

                exp_part[m*nu + k] = (cos(r_mc.dot(uv))+ I*sin(r_mc.dot(uv))) / 
                        (uv.dot(uv));
            }
        }
        //TSTOP(moo);

        // Now loop through an do the actual calculation.

        for (int j = 0; j < 3; j++) {
            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = d.triangles[i + (j+1)%3];
            Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

            std::size_t i_rn = d.triangles[i + j];
            Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

            std::size_t i_rn_1 = d.triangles[i + (j+2)%3];
            Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 3> ln1 = rn_1 - rn1;

            // Now loop through the UV points and calculate the Fourier
            // Transform.

            Vector<double, 3> zhat_cross_ln1 = zhat.cross(ln1);

            for (int m = 0; m < 3; m++) {
                for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                    integral[k] += ((bessel0_prefix1[m*nu+k] + 
                            I*bessel0_prefix2[m*nu+k] - 
                            I*bessel0_prefix3[m*nu+k]*rn1 - 
                            bessel0_prefix4[m*nu+k]) * bessel0[m*nu+k] - 
                            bessel1[m*nu+k]) * exp_part[m*nu+k];
                }
            }

            // Finally put into the real and imaginary components, and clean
            // out the integral array.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                std::size_t idy = k * nv;

                for (std::size_t l = 0; l < (std::size_t) nv; l++) {
                    vis_real[idy+l] += (flux[d.triangles[i+j]*nv+l] * 
                            zhat_cross_ln1.dot(integral[k]) / (2.*Area)).real();
                    vis_imag[idy+l] += (flux[d.triangles[i+j]*nv+l] * 
                            zhat_cross_ln1.dot(integral[k]) / (2.*Area)).imag();

                }

                integral[k] = 0.;
            }
        }
    }
    //printf("%f\n", TGIVE(moo));

    // Do the centering of the data.

    Vector<double, 2> center(-dx, -dy);

    //TCLEAR(moo); TSTART(moo);
    for (std::size_t i = 0; i < (std::size_t) nu; i++) {
        Vector <double, 2> uv(2*pi*u[i], 2*pi*v[i]);

        for (std::size_t j = 0; j < (std::size_t) nv; j++) {

            double vis_real_temp = vis_real[i*nv+j]*cos(center.dot(uv)) - 
                vis_imag[i*nv+j]*sin(center.dot(uv));
            double vis_imag_temp = vis_real[i*nv+j]*sin(center.dot(uv)) + 
                vis_imag[i*nv+j]*cos(center.dot(uv));

            vis_real[i*nv+j] = vis_real_temp;
            vis_imag[i*nv+j] = vis_imag_temp;
        }
    }
    //TSTOP(moo);
    //printf("%f\n", TGIVE(moo));

    // Clean up.

    delete[] integral; delete[] bessel1; delete[] bessel0_prefix1;
    delete[] bessel0_prefix2; delete[] bessel0_prefix3; 
    delete[] bessel0_prefix4; delete[] bessel0; delete[] exp_part;

}
