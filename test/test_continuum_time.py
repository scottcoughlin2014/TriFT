#!/usr/bin/env python3

import pdspy.interferometry as uv
import pdspy.modeling as modeling
import pdspy.dust as dust
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy
import trift
import time

################################################################################
#
# Run the model with RADMC3D
#
################################################################################

m = modeling.Model()

# Read in the opacities.

data = numpy.loadtxt('data/dustkappa_yso.inp', skiprows=2)

lam = data[:,0].copy() * 1.0e-4
kabs = data[:,1].copy()
ksca = data[:,2].copy()

d = dust.Dust()
d.set_properties(lam, kabs, ksca)

# Set up the grid.

nr = 100
nt = 100
np = 2

r = numpy.logspace(-1.,2.,100)
t = numpy.arange(nt)/(nt-1.)*numpy.pi/2
p = numpy.arange(np)/(np-1.)*2*numpy.pi

m.grid.set_spherical_grid(r, t, p)

# Set up the wavelength grid.

m.grid.lam = lam * 1.0e4

# Set the density.

dens = numpy.zeros((nr-1,nt-1,np-1)) + 1.0e-19

m.grid.add_density(dens, d)

# Add a star.

source = modeling.Star(mass=0.5, luminosity=0.22658222, temperature=4000.)

m.grid.add_star(source)

# Run the thermal simulation.

m.run_thermal(code="radmc3d", nphot=1e6, modified_random_walk=True, \
        verbose=False)

# Run the image.

t1 = time.time()

m.run_image(name="image", nphot=1e5, npix=100, pixelsize=0.025, lam="1000", \
        phi=0, incl=0, code="radmc3d", dpc=100, verbose=False, \
        unstructured=True, camera_nrrefine=6)

print(m.images["image"].x.size**0.5)

# Do the Fourier transform with TrIFT

u, v = numpy.meshgrid(numpy.linspace(-1.e6,1.e6,100), \
        numpy.linspace(-1.e6,1.e6,100))

u = u.reshape((-1,))
v = v.reshape((-1,))

vis = uv.interpolate_model(u, v, numpy.array([2.3e11]), m.images["image"], \
        code="trift", dRA=0., dDec=0.)
t2 = time.time()

triang = tri.Triangulation(m.images["image"].x, m.images["image"].y)
trift_res = numpy.inf
for triangle in triang.triangles:
    for i in range(3):
        length = numpy.sqrt((m.images["image"].x[triangle[i]] - \
                m.images["image"].x[triangle[i-1]])**2 + \
                (m.images["image"].y[triangle[i]] - \
                m.images["image"].y[triangle[i-1]])**2)
        if length < trift_res:
            trift_res = length
print(trift_res)

trift_npix = numpy.sqrt(m.images["image"].x.size)
trift_time = t2 - t1

# Get the time for each number of pixels.

times = []
npix_arr = [32,64,128,256,512,1024,2048]
pixelsize_arr = [25/npix for npix in npix_arr]

for npix in npix_arr:
    t1 = time.time()
    m.run_visibilities(name="image"+str(npix), nphot=1e5, npix=npix, \
            pixelsize=25./npix, lam="1000", phi=0, incl=0, code="radmc3d", \
            dpc=100, verbose=False, nostar=True)
    t2 = time.time()

    times.append(t2 - t1)

    m.visibilities["image"+str(npix)] = uv.average(m.visibilities["image"+\
            str(npix)], gridsize=1000000, binsize=1000, radial=True)

# Now make the plot.

plt.loglog(npix_arr, times, "k.-", label="Traditional Image")
plt.plot(trift_npix, trift_time, "o", label="Unstructured Image")

plt.xlabel("$\sqrt{N_{pix}}$", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)

plt.legend()

plt.gca().tick_params(labelsize=14)

plt.subplots_adjust(right=0.98, top=0.98, bottom=0.15, left=0.15)

plt.gcf().set_size_inches((4.5,4))

plt.show()

# Also make the plot for time vs. effective resolution.

plt.loglog(pixelsize_arr, times, "k.-", label="Traditional Image")
plt.plot(trift_res, trift_time, "o", label="Unstructured Image")

plt.xlabel('Effective Resolution ["]', fontsize=14)
plt.ylabel("Time (s)", fontsize=14)

plt.legend()

plt.gca().tick_params(labelsize=14)

plt.subplots_adjust(right=0.95, top=0.98, bottom=0.15, left=0.15)

plt.gcf().set_size_inches((4.5,4))

plt.show()

# Finally, plot the visibilities.

plt.loglog(u/1e3, vis.amp*1000, "k-", \
        label="Unstructured Fourier Transform, $N_{pix} = 165^2$")

"""
for npix in npix_arr:
    good = numpy.nonzero(m.visibilities["image"+str(npix)].real[:,0])
    plt.plot(m.visibilities["image"+str(npix)].u[good]/1e3, \
            m.visibilities["image"+str(npix)].amp[:,0][good]*1000, "-", \
            label="Traditional Image, $N_{{pix}} = {0:d}^2$".format(npix))
"""

plt.xlabel("u [k$\lambda$]", fontsize=14)
plt.ylabel(r"F$_{\nu}$ [mJy]", fontsize=14)

plt.legend(loc="lower left")

plt.axes().tick_params(labelsize=14)

plt.subplots_adjust(left=0.16, right=0.95, top=0.98, bottom=0.15)

plt.xlim(1.,14000.)

plt.show()
