#!/usr/bin/env python3

from pdspy.constants.astronomy import arcsec
import pdspy.interferometry as uv
import pdspy.modeling as modeling
import pdspy.dust as dust
import pdspy.gas as gas
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy
import trift
import time
import sys

################################################################################
#
# Run the model with RADMC3D
#
################################################################################

# Read in the opacities.

data = numpy.loadtxt('data/dustkappa_yso.inp', skiprows=2)

lam = data[:,0].copy() * 1.0e-4
kabs = data[:,1].copy()
ksca = data[:,2].copy()

d = dust.Dust()
d.set_properties(lam, kabs, ksca)

# Read in the gas information.

g = gas.Gas()
g.set_properties_from_lambda(gas.__path__[0]+"/data/co.dat")

# Generate the model.

m = modeling.YSOModel()

m.add_star(mass=0.5, luminosity=1.0, temperature=4000.)
m.set_spherical_grid(0.1, 1000., 100, 100, 2, code="radmc3d")

m.add_pringle_disk(mass=0.0001, rmin=0.1, rmax=50., plrho=2., h0=0.1, plh=1.0, \
        dust=d, t0=100, plt=0.25, gas=[g], abundance=[1.0e-4], aturb=0.1)

m.add_ulrich_envelope(mass=0.000001, rmin=0.1, rmax=1000., rcent=50, cavpl=1., \
        cavrfact=1., dust=d, t0=100, tpl=0.25, gas=[g], abundance=[1.0e-4], \
        aturb=0.1)

m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

# Run the image.

m.run_image(name="image", nphot=1e5, npix=25, pixelsize=1.0, lam=None, \
        tgas_eq_tdust=True, scattering_mode_max=0, incl_dust=False, \
        incl_lines=True, imolspec=1, iline=2, widthkms=10., linenlam=25, \
        pa=0, incl=45, code="radmc3d", dpc=100, verbose=False, \
        unstructured=True, camera_nrrefine=100, camera_refine_criterion=1, \
        nostar=True)

print(m.images["image"].image.shape)

npix_trift = int(m.images["image"].image[:,0].size**0.5)

# Plot the image.

triang = tri.Triangulation(m.images["image"].x, \
        m.images["image"].y)

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10,9), \
        gridspec_kw=dict(left=0.1, right=0.98, top=0.98, bottom=0.07, \
        hspace=0, wspace=0))

for i, ax in enumerate(axes.flatten()):
    ax.tripcolor(triang, m.images["image"].image[:,i], "ko-")
    #plt.triplot(triang, "k.-", linewidth=0.1, markersize=0)

    ax.set_aspect("equal")

    ax.set_xlim(10,-10)
    ax.set_ylim(-10,10)

    if i >= 20:
        ax.set_xlabel('$\Delta$ R.A. ["]', fontsize=14)
    else:
        ax.xaxis.set_ticklabels([])

    if i%5 == 0:
        ax.set_ylabel('$\Delta$ Dec. ["]', fontsize=14)
    else:
        ax.yaxis.set_ticklabels([])

    ax.tick_params(labelsize=14)

plt.show()

# Do the Fourier transform with TrIFT

ruv = numpy.linspace(100.,1.5e7,10000)

pa = 0.
u = ruv * numpy.cos(pa)
v = ruv * numpy.sin(pa)

t1 = time.time()
vis = uv.interpolate_model(u, v, m.images["image"].freq, m.images["image"], \
        code="trift", dRA=0., dDec=0.)
t2 = time.time()
print(t2 - t1)

# Do a model with GALARIO to compare with.

npix = 1024
imsize = 25.

t1 = time.time()
m.run_image(name="galario", nphot=1e5, npix=npix, pixelsize=imsize/npix,\
        lam=None, tgas_eq_tdust=True, scattering_mode_max=0, incl_dust=False, \
        incl_lines=True, imolspec=1, iline=2, widthkms=10., linenlam=25, \
        pa=0, incl=45, code="radmc3d", dpc=100, verbose=False, nostar=True)
t2 = time.time()
print(t2 - t1)

ruv = numpy.linspace(1./(imsize*arcsec),1./(2*imsize/npix*arcsec),10000)

u = ruv * numpy.cos(pa)
v = ruv * numpy.sin(pa)

vis_galario = uv.interpolate_model(u, v, m.images["galario"].freq, \
        m.images["galario"], code="galario", dRA=0., dDec=0.)

# Finally, plot the visibilities.

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10,9), \
        gridspec_kw=dict(left=0.1, right=0.98, top=0.92, bottom=0.07, \
        hspace=0, wspace=0))

for i, ax in enumerate(axes.flatten()):
    ax.loglog(vis.u/1e3, vis.amp[:,i]*1000, "k-")

    ax.plot(vis_galario.u/1e3, vis_galario.amp[:,i]*1000, "r-", \
            label="Traditional Image, $N_{pix} = 1024^2$")

    if i >= 20:
        ax.set_xlabel("u [k$\lambda$]", fontsize=14)
    else:
        ax.xaxis.set_ticklabels([])

    if i%5 == 0:
        ax.set_ylabel(r"F$_{\nu}$ [mJy]", fontsize=14)
    else:
        ax.yaxis.set_ticklabels([])

    ax.tick_params(labelsize=14)

fig.legend(loc="upper center", ncol=2, fontsize=14, \
        handles=axes[0,0].get_lines(), labels=("Unstructured Fourier "
        "Transform, $N_{{pix}} = {0:d}^2$".format(npix_trift), \
        "Traditional Image, $N_{pix} = 1024^2$"))

plt.show()
