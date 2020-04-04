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
