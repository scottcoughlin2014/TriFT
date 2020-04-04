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

r = numpy.logspace(-1.,3.,nr)
t = numpy.arange(nt)/(nt-1.)*numpy.pi/2
p = numpy.arange(np)/(np-1.)*2*numpy.pi

m.grid.set_spherical_grid(r, t, p)

# Set up the wavelength grid.

m.grid.lam = lam * 1.0e4

# Set the density.

rr, tt, pp = numpy.meshgrid(m.grid.r, m.grid.theta, m.grid.phi, indexing='ij')

dens = 1.0e-19 * (rr / rr.min())**-1.5 * numpy.exp(-0.5*(rr/750.)**4)

m.grid.add_density(dens, d)

# Add a star.

source = modeling.Star(mass=0.5, luminosity=0.22658222, temperature=4000.)

m.grid.add_star(source)

# Run the thermal simulation.

t1 = time.time()
m.run_thermal(code="radmc3d", nphot=1e6, modified_random_walk=True, \
        verbose=False, setthreads=2)
t2 = time.time()
print(t2-t1)

# Run the image.

m.run_image(name="image", nphot=1e5, npix=25, pixelsize=1.0, lam="1000", \
        phi=0, incl=0, code="radmc3d", dpc=100, verbose=False, \
        unstructured=True, camera_nrrefine=8, camera_refine_criterion=1, \
        nostar=True)

print(m.images["image"].image.shape)

# Plot the image.

triang = tri.Triangulation(m.images["image"].x, m.images["image"].y)

plt.tripcolor(triang, m.images["image"].image[:,0], "ko-")
plt.triplot(triang, "k.-", linewidth=0.1, markersize=0.1)

plt.axes().set_aspect("equal")

plt.xlim(-12,12)
plt.ylim(-12,12)

plt.xlabel('$\Delta$ R.A. ["]', fontsize=14)
plt.ylabel('$\Delta$ Dec. ["]', fontsize=14)

plt.gca().tick_params(labelsize=14)

plt.show()

# Do the Fourier transform with TrIFT

u = numpy.linspace(100.,1.5e7,10000)
v = numpy.repeat(0.001,10000)

t1 = time.time()
vis = uv.interpolate_model(u, v, numpy.array([2.3e11]), m.images["image"], \
        code="trift", dRA=0., dDec=0.)
t2 = time.time()
print(t2 - t1)

# Do a high resolution model to compare with.

m.run_visibilities(name="image", nphot=1e5, npix=1024, \
        pixelsize=8*25./1024, lam="1000", phi=0, incl=0, code="radmc3d", \
        dpc=100, verbose=False, nostar=True)

m.visibilities["image"] = uv.average(m.visibilities["image"], \
        gridsize=1000000, binsize=1000, radial=True)

# Finally, plot the visibilities.

plt.loglog(u/1e3, vis.amp*1000, "k-", \
        label="Unstructured Fourier Transform, $N_{pix} = 165^2$")

plt.plot(m.visibilities["image"].u/1e3, m.visibilities["image"].amp*1000, \
        "r-", label="Traditional Image, $N_{pix} = 1024^2$")

plt.xlabel("u [k$\lambda$]", fontsize=14)
plt.ylabel(r"F$_{\nu}$ [mJy]", fontsize=14)

plt.legend(loc="lower left")

plt.axes().tick_params(labelsize=14)

plt.subplots_adjust(left=0.16, right=0.95, top=0.98, bottom=0.15)

plt.xlim(1.,14000.)

plt.show()
