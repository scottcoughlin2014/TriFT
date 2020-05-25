#!/usr/bin/env python3

from galario import double
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.special
import pyDOE
import numpy
import trift
import time

# Make an image.

grid = pyDOE.lhs(2, samples=400)

r = grid[:,0] * 2
phi = grid[:,1] * 2*numpy.pi

grid = pyDOE.lhs(2, samples=2000)

r = numpy.hstack((r, grid[:,0]*0.04 + 0.98))
phi = numpy.hstack((phi, grid[:,1]*2*numpy.pi))

x = r * numpy.cos(phi) * numpy.cos(numpy.pi/3)
y = r * numpy.sin(phi)

flux = numpy.where(r < 1., y - y.min(), 0.)

pa = numpy.pi/4

xp = x * numpy.cos(-pa) - y * numpy.sin(-pa)
yp = x * numpy.sin(-pa) + y * numpy.cos(-pa)

x = xp
y = yp

# Also make a traditional image to compare with.

xx, yy = numpy.meshgrid(numpy.linspace(-15.,15.,1024), \
        numpy.linspace(-15.,15.,1024))

xp = xx * numpy.cos(-pa) - yy * numpy.sin(-pa)
yp = xx * numpy.sin(-pa) + yy * numpy.cos(-pa)

rr = numpy.sqrt((xp/numpy.cos(numpy.pi/3))**2 + yp**2)

fflux = numpy.where(rr < 1., yp - y.min(), 0.)

# Plot the image.

triang = tri.Triangulation(x, y)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

ax[0].tripcolor(triang, flux, "ko-")
ax[0].triplot(triang, "k.-", linewidth=0.1, markersize=0.1)

#ax[1].scatter(xx, yy, c=fflux, marker=".")
ax[1].imshow(fflux, origin="lower", interpolation="nearest")

for i in range(1):
    ax[i].set_aspect("equal")

    ax[i].set_xlim(1.1,-1.1)
    ax[i].set_ylim(-1.1,1.1)

    ax[i].set_xlabel("x", fontsize=14)
    ax[i].set_ylabel("y", fontsize=14)

    ax[i].tick_params(labelsize=14)

plt.show()

# Do the Fourier transform with TrIFT

u, v = numpy.meshgrid(numpy.linspace(-3.,3.,100),numpy.linspace(-3.,3.,100))

u = u.reshape((u.size,))
v = v.reshape((v.size,))

vis = trift.trift_cextended(x, y, flux, u, v, 0., 0.)

# Do the Fourier transform with GALARIO.

dxy = xx[0,1] - xx[0,0]

vvis = double.sampleImage(fflux, dxy, u, v, origin="lower")

# NOTE: GALARIO appears to be off by the complex conjugate, possibly because
# CASA spits out visibilities that need to be complex conjugated to have the 
# proper orientation (see note by Urvashi Rau, relayed by Ian through
# MPoL or Ryan through vis_sample.

vvis = numpy.conj(vvis)

# And do just a standard, numpy fft.

vvvis = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(fflux[:,::-1])))

uu = numpy.fft.fftshift(numpy.fft.fftfreq(1024, dxy))
vv = numpy.fft.fftshift(numpy.fft.fftfreq(1024, dxy))

uu, vv = numpy.meshgrid(uu, vv)

uu = uu.reshape((uu.size,))
vv = vv.reshape((vv.size,))
vvvis = vvvis.reshape((vvvis.size,))

# Finally, plot the visibilities.

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

ax[0,0].scatter(u, v, c=vis.real, marker=".")
ax[0,1].scatter(u, v, c=vis.imag, marker=".")

ax[1,0].scatter(u, v, c=vvis.real, marker=".")
ax[1,1].scatter(u, v, c=vvis.imag, marker=".")

"""
ax[1,0].scatter(uu, vv, c=vvvis.real, marker=".")
ax[1,1].scatter(uu, vv, c=vvvis.imag, marker=".")
"""

for a in ax.flatten():
    a.set_xlabel("u", fontsize=14)
    a.set_ylabel("v", fontsize=14)

    a.tick_params(labelsize=14)

plt.show()
