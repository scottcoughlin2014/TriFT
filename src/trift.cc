#include "trift.h"
#include <delaunator.hpp>
#include "timer.c"
#include "fastmath.h"

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
                    (ln.dot(uv) * ln_1.dot(uv)) * FastCos(rn_dot_uv);
                vis_imag[k] += intensity_triangle * ln_1_dot_zhat_cross_ln /
                    (ln.dot(uv) * ln_1.dot(uv)) * FastSin(rn_dot_uv);
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

    Vector<std::complex<double>, 3> *integral_part1 = new 
            Vector<std::complex<double>,3>[nu];
    std::complex<double> *integral_part2 = new std::complex<double>[nu];

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

                // The various parts of the long integral equation, just
                // split up for ease of reading.

                Vector<double, 3> bessel0_prefix1 = zhat_cross_lm;
                Vector<double, 3> bessel0_prefix2 = r_mc * zhat_dot_lm_cross_uv;
                double bessel0_prefix3 = zhat_dot_lm_cross_uv;
                Vector<double, 3> bessel0_prefix4 = 2.*uv/uv.dot(uv)*
                        zhat_dot_lm_cross_uv;

                double bessel0 = BesselJ0(uv.dot(lm)/2.);

                Vector<double, 3> bessel1 = lm * (zhat_dot_lm_cross_uv/2.) * 
                        BesselJ1(uv.dot(lm)/2.);

                std::complex<double> exp_part = (FastCos(r_mc.dot(uv))+ 
                        I*FastSin(r_mc.dot(uv))) / (uv.dot(uv));

                // Now add everything together.

                integral_part1[k] += ((bessel0_prefix1 + I*bessel0_prefix2 - 
                        bessel0_prefix4) * bessel0 - bessel1) * exp_part;
                integral_part2[k] += -I*bessel0_prefix3 * bessel0 * exp_part;
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

            Vector<double, 3> zhat_cross_ln1 = zhat.cross(ln1);

            // Finally put into the real and imaginary components.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                vis_real[k] += (intensity * zhat_cross_ln1.dot(
                        integral_part1[k]+rn1*integral_part2[k]) / 
                        (2.*Area)).real();
                vis_imag[k] += (intensity * zhat_cross_ln1.dot(
                        integral_part1[k]+rn1*integral_part2[k]) / 
                        (2.*Area)).imag();
            }
        }

        // Clear out the integral array for the next triangle.

        for (std::size_t k = 0; k < (std::size_t) nu; k++) {
            integral_part1[k] = 0.;
            integral_part2[k] = 0.;
        }
    }
    //printf("%f\n", TGIVE(moo));

    // Clean up.

    delete[] integral_part1; delete [] integral_part2;

    // Do the centering of the data.

    Vector<double, 2> center(-dx, -dy);

    for (std::size_t i = 0; i < (std::size_t) nu; i++) {
        Vector <double, 2> uv(2*pi*u[i], 2*pi*v[i]);

        double vis_real_temp = vis_real[i]*cos(center.dot(uv)) - vis_imag[i]*
            sin(center.dot(uv));
        double vis_imag_temp = vis_real[i]*sin(center.dot(uv)) + vis_imag[i]*
            cos(center.dot(uv));

        vis_real[i] = vis_real_temp;
        vis_imag[i] = vis_imag_temp;
    }
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
                        FastCos(rn_dot_uv);
                    vis_imag[idy+l] += intensity_triangle[l] * 
                        ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                        FastSin(rn_dot_uv);
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

    Vector<std::complex<double>, 3> *integral_part1 = new 
            Vector<std::complex<double>,3>[nu];
    std::complex<double> *integral_part2 = new std::complex<double>[nu];

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

                // The various parts of the long integral equation, just
                // split up for ease of reading.

                Vector<double, 3> bessel0_prefix1 = zhat_cross_lm;
                Vector<double, 3> bessel0_prefix2 = r_mc * zhat_dot_lm_cross_uv;
                double bessel0_prefix3 = zhat_dot_lm_cross_uv;
                Vector<double, 3> bessel0_prefix4 = 2.*uv/uv.dot(uv)*
                        zhat_dot_lm_cross_uv;

                double bessel0 = BesselJ0(uv.dot(lm)/2.);

                Vector<double, 3> bessel1 = lm * (zhat_dot_lm_cross_uv/2.) * 
                        BesselJ1(uv.dot(lm)/2.);

                std::complex<double> exp_part = (FastCos(r_mc.dot(uv))+ 
                        I*FastSin(r_mc.dot(uv))) / (uv.dot(uv));

                // Now add everything together.

                integral_part1[k] += ((bessel0_prefix1 + I*bessel0_prefix2 - 
                        bessel0_prefix4) * bessel0 - bessel1) * exp_part;
                integral_part2[k] += -I*bessel0_prefix3 * bessel0 * exp_part;
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

            Vector<double, 3> zhat_cross_ln1 = zhat.cross(ln1);

            // Finally put into the real and imaginary components.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                std::size_t idy = k * nv;

                for (std::size_t l = 0; l < (std::size_t) nv; l++) {
                    vis_real[idy+l] += (flux[d.triangles[i+j]*nv+l] * 
                            zhat_cross_ln1.dot(integral_part1[k]+rn1*
                            integral_part2[k]) / (2.*Area)).real();
                    vis_imag[idy+l] += (flux[d.triangles[i+j]*nv+l] * 
                            zhat_cross_ln1.dot(integral_part1[k]+rn1*
                            integral_part2[k]) / (2.*Area)).imag();
                }
            }
        }

        // Clear out the integral arrays for the next triangle.

        for (std::size_t k = 0; k < (std::size_t) nu; k++) {
            integral_part1[k] = 0.;
            integral_part2[k] = 0.;
        }
    }
    //printf("%f\n", TGIVE(moo));

    // Clean up.

    delete[] integral_part1; delete [] integral_part2;

    // Do the centering of the data.

    Vector<double, 2> center(-dx, -dy);

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
}
