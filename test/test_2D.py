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

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

triang = tri.Triangulation(x, y)

for i in range(2):
    ax[i].tripcolor(triang, flux[:,i], "ko-")
    ax[i].triplot(triang, "k.-", linewidth=0.1, markersize=0.1)

    ax[i].set_aspect("equal")

    ax[i].set_xlim(-1.5,1.5)
    ax[i].set_ylim(-1.5,1.5)

    ax[i].set_xlabel("x", fontsize=14)
    ax[i].set_ylabel("y", fontsize=14)

    ax[i].tick_params(labelsize=14)

fig.subplots_adjust(wspace=0.3)

plt.show()

# Do the Fourier transform with TrIFT

u = numpy.linspace(0.001,10.,1000)
v = numpy.repeat(0., 1000)

t1 = time.time()
vis = trift.trift_2D(x, y, flux, u, v, 0.25, 0.25, nthreads=4)
t2 = time.time()
print(t2 - t1)

#t1 = time.time()
#vis_extended = trift.trift_2Dextended(x, y, flux, u, v, 0.25, 0.25, nthreads=4)
#t2 = time.time()
#print(t2 - t1)

# Calculate the analytic result.

vis_analytic = numpy.zeros(vis.shape, dtype=complex)

vis_analytic[:,0] = scipy.special.jv(1, 2*numpy.pi*u) / u * \
        numpy.exp(2*numpy.pi*1j*(0.25*u + 0.25*v))

vis_analytic[:,1] = 4.*numpy.sinc(2*u) * numpy.sinc(2*v) * \
        numpy.exp(2*numpy.pi*1j*(0.25*u + 0.25*v))

# Finally, plot the visibilities.

fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

for i in range(2):
    ax[i].plot(u, vis.real[:,i], "k.-", label="Unstructured Fourier Transform")
    ax[i].plot(u, vis_extended.real[:,i], "b.-", label="Unstructured Fourier "
            "Transform, Extended Version")
    ax[i].plot(u, vis_analytic.real[:,i], "r-", label="Analytic Solution")

    ax[i].set_xlabel("u", fontsize=14)
    ax[i].set_ylabel("Real Component", fontsize=14)

    ax[i].tick_params(labelsize=14)

#fig.subplots_adjust(left=0.05, bottom=0.45, right=0.98, top=0.99, wspace=0.5)

ax[0].legend(fontsize=10, loc="upper right")

plt.show()
