#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.special
import pyDOE
import numpy
import trift
import time

# Make an image.

grid = pyDOE.lhs(2, samples=30000)

x = grid[:,0] * 4 - 2.
y = grid[:,1] * 4 - 2.

r = (x**2 + y**2)**0.5

flux = numpy.zeros((r.size,2))

flux[:,0] = numpy.where(r < 1., 1., 0.)
flux[:,1] = numpy.where(numpy.logical_and(numpy.abs(x) < 1, numpy.abs(y) < 1), \
        1., 0.)

# Plot the image.

triang = tri.Triangulation(x, y)

plt.tripcolor(triang, flux[:,1], "ko-")
plt.show()

# Do the Fourier transform with TrIFT

u = numpy.linspace(0.001,10.,1000)
v = numpy.repeat(0., 1000)

t1 = time.time()
vis = trift.trift_2D(x, y, flux, u, v, 0.25, 0.25)
t2 = time.time()
print(t2 - t1)

# Calculate the analytic result.

vis_analytic = numpy.zeros(vis.shape, dtype=complex)

vis_analytic[:,0] = scipy.special.jv(1, 2*numpy.pi*u) / u * \
        numpy.exp(2*numpy.pi*1j*(0.25*u + 0.25*v))

vis_analytic[:,1] = 4.*numpy.sinc(2*u) * numpy.sinc(2*v) * \
        numpy.exp(2*numpy.pi*1j*(0.25*u + 0.25*v))

# Finally, plot the visibilities.

for i in range(2):
    plt.plot(u, vis.real[:,i], "b.-")
    plt.plot(u, vis_analytic.real[:,i], "r-")
    plt.show()
