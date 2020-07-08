#include "trift.h"
#include <delaunator-header-only.hpp>
#include "timer.c"
#include "fastmath.h"
#include <unordered_map>
#include "cuda_complex.hpp"


#ifdef __CUDACC__
#else
#include "omp.h"
#endif


#ifdef __GLOBAL_FIT__
#ifdef __CUDACC__
__device__
#endif // __CUDACC__
#else
CUDA_CALLABLE_MEMBER
#endif // __GLOBAL_FIT__

#ifdef __CUDACC__
    CUDA_KERNEL

    void trift2D(double *x, double *y, double *flux, double *u, double *v,
            double *vis_real, double *vis_imag, int nx, int nu, int nv,
            double dx, double dy, int nthreads) {

        // Use only 1 thread first, otherwise Delaunator could have a segfault.

        omp_set_num_threads(1);

        // Set up the coordinates for the triangulation.
        
        std::vector<double> coords;

        for (int i=0; i < nx; i++) {
            coords.push_back(x[i]);
            coords.push_back(y[i]);
        }

        // Run the Delauney triangulation here.

        delaunator::Delaunator d(coords);

        // Now set to the appropriate number of threads for the remainder of the 
        // program.

        omp_set_num_threads(nthreads);

        // Loop through and take the Fourier transform of each triangle.
        
        Vector<double, 3> zhat(0., 0., 1.);

        double **vis_real_tmp = new double*[nthreads];
        double **vis_imag_tmp = new double*[nthreads];
        for (std::size_t i = 0; i < (std::size_t) nthreads; i++) {
            vis_real_tmp[i] = new double[nu*nv];
            vis_imag_tmp[i] = new double[nu*nv];
        }

        #pragma omp parallel
        {
        int thread_id = omp_get_thread_num();

        for (std::size_t i = 0; i < (std::size_t) nu*nv; i++) {
            vis_real_tmp[thread_id][i] = 0;
            vis_imag_tmp[thread_id][i] = 0;
        }

        double *intensity_triangle = new double[nv];

        #pragma omp for
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
                        vis_real_tmp[thread_id][idy+l] += intensity_triangle[l] * 
                            ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                            FastCos(rn_dot_uv);
                        vis_imag_tmp[thread_id][idy+l] += intensity_triangle[l] * 
                            ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                            -FastSin(rn_dot_uv);
                    }
                }
            }
        }
        delete[] intensity_triangle;
        }

        // Now add together all of the separate vis'.

        #pragma omp parallel for
        for (std::size_t i = 0; i < (std::size_t) nu*nv; i++) {
            for (std::size_t j = 0; j < (std::size_t) nthreads; j++) {
                vis_real[i] += vis_real_tmp[j][i];
                vis_imag[i] += vis_imag_tmp[j][i];
            }
        }

        // And clean up the tmp arrays.
        
        for (std::size_t i = 0; i < (std::size_t) nthreads; i++) {
            delete[] vis_real_tmp[i]; delete[] vis_imag_tmp[i];
        }
        delete[] vis_real_tmp; delete[] vis_imag_tmp;

        // Do the centering of the data.

        Vector<double, 2> center(-dx, -dy);

        //TCLEAR(moo); TSTART(moo);
        #pragma omp parallel for collapse(2)
        for (std::size_t i = 0; i < (std::size_t) nu; i++) {
            for (std::size_t j = 0; j < (std::size_t) nv; j++) {
                Vector <double, 2> uv(2*pi*u[i], 2*pi*v[i]);

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
    }

#else
    void trift2D(double *x, double *y, double *flux, double *u, double *v,
            double *vis_real, double *vis_imag, int nx, int nu, int nv,
            double dx, double dy, int nthreads) {

        // Use only 1 thread first, otherwise Delaunator could have a segfault.

        omp_set_num_threads(1);

        // Set up the coordinates for the triangulation.
        
        std::vector<double> coords;

        for (int i=0; i < nx; i++) {
            coords.push_back(x[i]);
            coords.push_back(y[i]);
        }

        // Run the Delauney triangulation here.

        delaunator::Delaunator d(coords);

        // Now set to the appropriate number of threads for the remainder of the 
        // program.

        omp_set_num_threads(nthreads);

        // Loop through and take the Fourier transform of each triangle.
        
        Vector<double, 3> zhat(0., 0., 1.);

        double **vis_real_tmp = new double*[nthreads];
        double **vis_imag_tmp = new double*[nthreads];
        for (std::size_t i = 0; i < (std::size_t) nthreads; i++) {
            vis_real_tmp[i] = new double[nu*nv];
            vis_imag_tmp[i] = new double[nu*nv];
        }

        #pragma omp parallel
        {
        int thread_id = omp_get_thread_num();

        for (std::size_t i = 0; i < (std::size_t) nu*nv; i++) {
            vis_real_tmp[thread_id][i] = 0;
            vis_imag_tmp[thread_id][i] = 0;
        }

        double *intensity_triangle = new double[nv];

        #pragma omp for
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
                        vis_real_tmp[thread_id][idy+l] += intensity_triangle[l] * 
                            ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                            FastCos(rn_dot_uv);
                        vis_imag_tmp[thread_id][idy+l] += intensity_triangle[l] * 
                            ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                            -FastSin(rn_dot_uv);
                    }
                }
            }
        }
        delete[] intensity_triangle;
        }

        // Now add together all of the separate vis'.

        #pragma omp parallel for
        for (std::size_t i = 0; i < (std::size_t) nu*nv; i++) {
            for (std::size_t j = 0; j < (std::size_t) nthreads; j++) {
                vis_real[i] += vis_real_tmp[j][i];
                vis_imag[i] += vis_imag_tmp[j][i];
            }
        }

        // And clean up the tmp arrays.
        
        for (std::size_t i = 0; i < (std::size_t) nthreads; i++) {
            delete[] vis_real_tmp[i]; delete[] vis_imag_tmp[i];
        }
        delete[] vis_real_tmp; delete[] vis_imag_tmp;

        // Do the centering of the data.

        Vector<double, 2> center(-dx, -dy);

        //TCLEAR(moo); TSTART(moo);
        #pragma omp parallel for collapse(2)
        for (std::size_t i = 0; i < (std::size_t) nu; i++) {
            for (std::size_t j = 0; j < (std::size_t) nv; j++) {
                Vector <double, 2> uv(2*pi*u[i], 2*pi*v[i]);

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
    }
#endif
