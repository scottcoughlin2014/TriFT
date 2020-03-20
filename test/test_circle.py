#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.special
import pyDOE
import numpy
import trift
import time

# Make an image.

grid = pyDOE.lhs(2, samples=100)

r = grid[:,0] * 2
phi = grid[:,1] * 2*numpy.pi

grid = pyDOE.lhs(2, samples=2000)

r = numpy.hstack((r, grid[:,0]*0.04 + 0.98))
phi = numpy.hstack((phi, grid[:,1]*2*numpy.pi))

x = r * numpy.cos(phi)
y = r * numpy.sin(phi)

flux = numpy.where(r < 1., 1., 0.)

# Plot the image.

triang = tri.Triangulation(x, y)

plt.tripcolor(triang, flux, "ko-")
plt.triplot(triang, "k.-", linewidth=0.1, markersize=0.1)
plt.show()

# Do the Fourier transform with TrIFT

u = numpy.linspace(0.001,10.,1000)
v = numpy.repeat(0., 1000)

t1 = time.time()
vis = trift.trift_c(x, y, flux, u, v, 0.25, 0.25, nthreads=4)
t2 = time.time()
print(t2 - t1)

# Calculate the analytic result.

vis_analytic = scipy.special.jv(1, 2*numpy.pi*u) / u * numpy.exp(2*numpy.pi*\
        1j*(0.25*u + 0.25*v))

# Finally, plot the visibilities.

plt.plot(u, vis.real, "k.-")
plt.plot(u, vis_analytic.real, "r-")
plt.show()
