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

t1 = time.time()
m.run_thermal(code="radmc3d", nphot=1e6, modified_random_walk=True, \
        verbose=False)
t2 = time.time()
print(t2-t1)

# Run the image.

m.run_image(name="image", nphot=1e5, npix=25, pixelsize=0.1, lam="1000", \
        phi=0, incl=0, code="radmc3d", dpc=100, verbose=True, \
        unstructured=True)

print(m.images["image"].image.shape)

# Plot the image.

triang = tri.Triangulation(m.images["image"].x, m.images["image"].y)

plt.tripcolor(triang, m.images["image"].image[:,0], "ko-")
plt.show()

# Do the Fourier transform with TrIFT

u, v = numpy.meshgrid(numpy.linspace(-1.e6,1.e6,100), \
        numpy.linspace(-1.e6,1.e6,100))

u = u.reshape((-1,))
v = v.reshape((-1,))

t1 = time.time()
"""
vis = trift.trift_c(m.images["image"].x, m.images["image"].y, \
        m.images["image"].image[:,0], u, v)
"""
vis = uv.interpolate_model(u, v, numpy.array([2.3e11]), m.images["image"], \
        code="trift")
t2 = time.time()
print(t2 - t1)

# Finally, plot the visibilities.

plt.imshow(vis.real[:,0].reshape((100,100)))
plt.show()
