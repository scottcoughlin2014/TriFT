#include "trift.h"
#include <delaunator-header-only.hpp>
#include "timer.c"
#include "fastmath.h"
#include <unordered_map>

void trift_precalc(double *x, double *y, double *flux, double *u, double *v,
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

    // Get the number of times each vertex is included.

    int *vertex_count = new int[nx];
    for (std::size_t i = 0; i < (std::size_t)nx; i++)
        vertex_count[i] = 0;
    for (std::size_t i = 0; i < d.triangles.size(); i++)
        vertex_count[d.triangles[i]]++;

    // Set up the maps for the saved cos and sin values.

    std::unordered_map<int,double*> CosMap;
    std::unordered_map<int,double*> SinMap;
    bool *vertex_calculated = new bool[nx];
    for (std::size_t i = 0; i < (std::size_t)nx; i++)
        vertex_calculated[i] = false;

    //int max_size = 0;

    // Set up the uv points array.

    Vector<double, 2> *uv = new Vector<double, 2>[nu];
    for (std::size_t i = 0; i < (std::size_t)nu; i++) {
        uv[i][0] = 2*pi*u[i]; uv[i][1] = 2*pi*v[i];
    }

    // Loop through and take the Fourier transform of each triangle.
    
    Vector<double, 3> zhat(0., 0., 1.);

    //TCREATE(moo); TCLEAR(moo); TSTART(moo);
    for (std::size_t i = 0; i < d.triangles.size(); i+=3) {
        double intensity_triangle = (flux[d.triangles[i]] + 
            flux[d.triangles[i+1]] + flux[d.triangles[i+2]]) / 3.;

        // Now, loop through and do the real calculation.

        for (int j = 0; j < 3; j++) {
            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = d.triangles[i + (j+1)%3];
            Vector<double, 2> rn1(x[i_rn1], y[i_rn1]);

            std::size_t i_rn = d.triangles[i + j];
            Vector<double, 2> rn(x[i_rn], y[i_rn]);

            std::size_t i_rn_1 = d.triangles[i + (j+2)%3];
            Vector<double, 2> rn_1(x[i_rn_1], y[i_rn_1]);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 2> ln = rn1 - rn;
            Vector<double, 2> ln_1 = rn - rn_1;

            // If this vertex hasn't been calculated, do so and save.

            if (not vertex_calculated[d.triangles[i+j]]) {
                CosMap[d.triangles[i+j]] = new double[nu];
                SinMap[d.triangles[i+j]] = new double[nu];

                double *CosArr = CosMap[d.triangles[i+j]];
                double *SinArr = SinMap[d.triangles[i+j]];
                for (std::size_t k = 0; k < (std::size_t)nu; k++) {
                    double RnDotUvArr = rn.dot(uv[k]);
                    CosArr[k] = FastCos(RnDotUvArr);
                    SinArr[k] = FastSin(RnDotUvArr);
                }

                vertex_calculated[d.triangles[i+j]] = true;
            }

            // Now loop through the UV points and calculate the Fourier
            // Transform.

            Vector<double, 2> zhat_cross_ln(-ln[1],ln[0]);
            double ln_1_dot_zhat_cross_ln = ln_1.dot(zhat_cross_ln);

            double *CosArr = CosMap[d.triangles[i+j]];
            double *SinArr = SinMap[d.triangles[i+j]];
            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                double temp = intensity_triangle * ln_1_dot_zhat_cross_ln /
                    (ln.dot(uv[k]) * ln_1.dot(uv[k]));

                vis_real[k] += temp * CosArr[k];
                vis_imag[k] += temp * SinArr[k];
            }
        }

        //if (CosMap.size() > (std::size_t)max_size)
        //    max_size = (int)CosMap.size();

        // Check whether this vertex is all used up, and if so, delete.

        for (int j = 0; j < 3; j++) {
            vertex_count[d.triangles[i+j]]--;

            if (vertex_count[d.triangles[i+j]] == 0) {
                delete[] CosMap[d.triangles[i+j]];
                delete[] SinMap[d.triangles[i+j]];
                CosMap.erase(d.triangles[i+j]);
                SinMap.erase(d.triangles[i+j]);
                vertex_calculated[d.triangles[i+j]] = false;
            }
        }
    }
    //printf("Final map size = %d\n", (int)CosMap.size());
    //printf("Max size = %d\n", max_size);
    //TSTOP(moo);
    //printf("%f\n", TGIVE(moo));
    
    delete[] vertex_calculated; delete[] vertex_count; delete[] uv;

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

void trift2D_precalc(double *x, double *y, double *flux, double *u, double *v,
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

    // Get the number of times each vertex is included.

    int *vertex_count = new int[nx];
    for (std::size_t i = 0; i < (std::size_t)nx; i++)
        vertex_count[i] = 0;
    for (std::size_t i = 0; i < d.triangles.size(); i++)
        vertex_count[d.triangles[i]]++;

    // Set up the maps for the saved cos and sin values.

    std::unordered_map<int,double*> CosMap;
    std::unordered_map<int,double*> SinMap;
    bool *vertex_calculated = new bool[nx];
    for (std::size_t i = 0; i < (std::size_t)nx; i++)
        vertex_calculated[i] = false;

    //int max_size = 0;

    // Set up the uv points array.

    Vector<double, 2> *uv = new Vector<double, 2>[nu];
    for (std::size_t i = 0; i < (std::size_t)nu; i++) {
        uv[i][0] = 2*pi*u[i]; uv[i][1] = 2*pi*v[i];
    }

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
            Vector<double, 2> rn1(x[i_rn1], y[i_rn1]);

            std::size_t i_rn = d.triangles[i + j];
            Vector<double, 2> rn(x[i_rn], y[i_rn]);

            std::size_t i_rn_1 = d.triangles[i + (j+2)%3];
            Vector<double, 2> rn_1(x[i_rn_1], y[i_rn_1]);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 2> ln = rn1 - rn;
            Vector<double, 2> ln_1 = rn - rn_1;

            // If this vertex hasn't been calculated, do so and save.

            if (not vertex_calculated[d.triangles[i+j]]) {
                CosMap[d.triangles[i+j]] = new double[nu];
                SinMap[d.triangles[i+j]] = new double[nu];

                double *CosArr = CosMap[d.triangles[i+j]];
                double *SinArr = SinMap[d.triangles[i+j]];
                for (std::size_t k = 0; k < (std::size_t)nu; k++) {
                    double RnDotUvArr = rn.dot(uv[k]);
                    CosArr[k] = FastCos(RnDotUvArr);
                    SinArr[k] = FastSin(RnDotUvArr);
                }

                vertex_calculated[d.triangles[i+j]] = true;
            }

            // Now loop through the UV points and calculate the Fourier
            // Transform.

            Vector<double, 2> zhat_cross_ln(-ln[1],ln[0]);
            double ln_1_dot_zhat_cross_ln = ln_1.dot(zhat_cross_ln);

            double *CosArr = CosMap[d.triangles[i+j]];
            double *SinArr = SinMap[d.triangles[i+j]];
            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                double temp = ln_1_dot_zhat_cross_ln / (ln.dot(uv[k]) * 
                        ln_1.dot(uv[k]));
                double temp_sin = temp * SinArr[k];
                double temp_cos = temp * CosArr[k];

                std::size_t idy = k * nv;
                for (std::size_t l = 0; l < (std::size_t) nv; l++) {
                    vis_real[idy+l] += intensity_triangle[l] * temp_cos;
                    vis_imag[idy+l] += intensity_triangle[l] * temp_sin;
                }
            }
        }

        //if (CosMap.size() > (std::size_t)max_size)
        //    max_size = (int)CosMap.size();

        // Check whether this vertex is all used up, and if so, delete.

        for (int j = 0; j < 3; j++) {
            vertex_count[d.triangles[i+j]]--;

            if (vertex_count[d.triangles[i+j]] == 0) {
                delete[] CosMap[d.triangles[i+j]];
                delete[] SinMap[d.triangles[i+j]];
                CosMap.erase(d.triangles[i+j]);
                SinMap.erase(d.triangles[i+j]);
                vertex_calculated[d.triangles[i+j]] = false;
            }
        }
    }
    //printf("Final map size = %d\n", (int)CosMap.size());
    //printf("Max size = %d\n", max_size);
    //TSTOP(moo);
    //printf("%f\n", TGIVE(moo));
    
    delete[] vertex_calculated; delete[] vertex_count; delete[] uv;


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

    // Clean up.

    delete[] intensity_triangle;
}
