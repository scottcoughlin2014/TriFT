#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.special
import pyDOE
import numpy
import trift
import time

# Make an image.

r, phi = numpy.meshgrid(numpy.hstack((numpy.linspace(0.,0.98,15), \
        numpy.linspace(0.98,1.02,30), numpy.linspace(1.02,2.,15))), \
        numpy.linspace(0.,2*numpy.pi,100))

r = r.reshape((r.size,))*numpy.random.uniform(0.999,1.001,r.size)
phi = phi.reshape((phi.size,))*numpy.random.uniform(0.999,1.001,phi.size)

x = r * numpy.cos(phi)
y = r * numpy.sin(phi)

flux = numpy.where(r < 1., 1., 0.)

# Plot the image.

triang = tri.Triangulation(x, y)

plt.tripcolor(triang, flux, "ko-")
plt.show()

# Do the Fourier transform with TrIFT

u = numpy.linspace(0.001,10.,1000)
v = numpy.repeat(0., 1000)

t1 = time.time()
vis = trift.trift_c(x, y, flux, u, v, 0.25, 0.25)
t2 = time.time()
print(t2 - t1)

# Calculate the analytic result.

vis_analytic = scipy.special.jv(1, 2*numpy.pi*u) / u * numpy.exp(2*numpy.pi*\
        1j*(0.25*u + 0.25*v))

# Finally, plot the visibilities.

plt.plot(u, vis.real, "k.-")
plt.plot(u, vis_analytic.real, "r-")
plt.show()
