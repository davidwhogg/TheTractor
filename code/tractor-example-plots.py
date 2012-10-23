'''
% DSTN:  This should be a 3-across by 2-down figure,
\caption{Demonstration of use of the profiles, or the implicit
  generative model in this \documentname.  \textsl{top-left:} The
  circular dimensionless $M^{\luv}=8$ mixture-of-Gaussian
  approximation to the luv profile, represented on a very fine pixel
  grid. \textsl{top-middle:} The ellipse representing the non-trivial
  affine transformation to be applied to the circular, dimensionless
  profile.  \textsl{top-right:} The sheared profile.
  \textsl{bottom-left:} A $K=3$ mixture-of-Gaussian model of the
  pixel-convolved point-spread function, represented on the very fine
  pixel grid.  \textsl{bottom-middle:} The sheared profile convolved
  with the PSF, represented on the very fine pixel grid.
  \textsl{bottom-right:} The sheared luv convolved with the PSF, but
  now shown on a realistic pixel grid.  Because by assumption the PSF
  is a pixel-convolved PSF, the representation on the realistic grid
  is found simply by interpolating the convolved high-resolution
  model.\label{fig:example}}
\end{figure}
'''

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from tractor import *
from tractor.sdss_galaxy import *
from astrometry.util.util import Tan
from astrometry.util.plotutils import antigray

from matplotlib.ticker import FixedFormatter

ps = PlotSequence('trdemo')

ra,dec = 0.,0.
rd = RaDecPos(ra, dec)
m1 = Flux(100.)
re = 1.
s1 = GalaxyShape(re, 1., 0.)
s2 = GalaxyShape(re, 0.5, 40.)
g1 = DevGalaxy(rd, m1, s1)

H,W = 501,501
print 'Size', W
data = np.zeros((H,W))
invvar = np.ones((H,W))
psf1 = NCircularGaussianPSF([1e-12], [1.])
psf1.minradius = 100.
nre = 3.
pixscale = re * 8. * nre / float(W) / 3600.
print 'Pixscale', pixscale
wcs1 = FitsWcs(Tan(ra, dec, W/2. + 0.5, H/2. + 0.5,
				   pixscale, 0., 0., pixscale, W, H))

im1 = Image(data=data, invvar=invvar, psf=psf1, wcs=wcs1,
			sky=ConstantSky(0.), photocal=NullPhotoCal(),
			name='im1')

tractor = Tractor([im1], [g1])

print 'getting initial model'
mod = tractor.getModelImage(0)
print 'gotit'

galext = [-nre*4,nre*4]*2
ima = dict(interpolation='nearest', origin='lower', cmap=antigray)

mod = np.maximum(mod, 1e-42)
logmrange = np.max(np.log10(mod)) - np.min(np.log10(mod))
logmrange = 6.

def plog(mod):
	plt.clf()
	lm = np.log10(mod + 1e-42)
	mx = lm.max()
	plt.imshow(lm-mx, vmax=0., vmin=-logmrange,
			   extent=galext, **ima)
	#plt.gray()
	ticks = np.arange(-6, 1)
	plt.colorbar(ticks=ticks,
				 format=FixedFormatter(
					 ['$10^{%i}$'%i for i in ticks[:-1]] + ['$1$']))
	ps.savefig()

plt.clf()
plt.imshow(mod, extent=galext, **ima)
plt.colorbar()
ax = plt.axis()
ps.savefig()

plog(mod)

angles = np.linspace(0., 2.*np.pi, 361)
i0 = np.argmin(np.abs(angles - np.deg2rad(180.)))
i90 = np.argmin(np.abs(angles - np.deg2rad(90.)))
u = np.cos(angles)
v = np.sin(angles)
G = s2.getRaDecBasis()
uv = np.vstack((u,v)).T
print uv.shape
print G.shape
dradec = np.dot(G, uv.T).T
print 'G', G
print dradec.shape
#S = 4.
#dradec *= S
dra  = dradec[:,0]
ddec = dradec[:,1]
# xy = np.array([wcs1.positionToPixel(RaDecPos(ra + idra, dec + iddec))
# 			   for idra,iddec in zip(dra,ddec)])
# xx,yy = xy[:,0], xy[:,1]
# 
# plt.clf()
# plt.plot(xx, yy, 'k-')
# plt.axis(ax)
# ps.savefig()

S = 3600.
plt.clf()
plt.plot(dra * S, ddec * S, 'k-')
for i in [i0, i90]:
	L = 0.3/4
	plt.arrow(0, 0, dra[i]*S, ddec[i]*S, shape='full',
			  overhang=0, head_length=L, head_width=0.5*L, fc='k',
			  length_includes_head=True)
plt.axvline(0., color='k', alpha=0.3)
plt.axhline(0., color='k', alpha=0.3)
plt.axis('equal')
ps.savefig()

mod0 = mod

g1.shape = s2
mod = tractor.getModelImage(0)

plog(mod)

t = np.deg2rad(70.)
R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
#o = np.array([[2.,0.],[0.,0.5]])
o = np.array([[2.,0.],[0.,0.3]])
V2 = np.dot(R.T, np.dot(o, R))
print 'V', V2

V1 = np.eye(2) * 3.
print 'V', V1

V3 = np.array([[4., -0.5],[-0.5,3.]])

VV = np.array([V1, V2, V3])
print 'VV', VV

psf2 = GaussianMixturePSF(np.array([0.6, 0.6, 0.6]),
						  np.array([[0.,0.], [-.5,0.], [1.,-0.5]]), VV)
im1.psf = psf2

p = psf2.getPointSourcePatch(0., 0.,)
print 'psf2 patch', p

ext0 = p.getExtent()
ext0 = [x-0.5 for x in ext0]
#print 'Extent', ext0
plt.clf()
plt.imshow(p.patch, extent=ext0, **ima)
#plt.axvline(0, color='k', alpha=0.5)
ax0 = [-10,10,-10,10]
plt.axis(ax0)
plt.colorbar()
ps.savefig()

plt.clf()
plt.imshow(np.log10(np.maximum(p.patch, 1e-10)), vmin=-10, 
		   extent=ext0, **ima)
plt.colorbar()
plt.axis(ax0)
ps.savefig()

#S = 25
S = 20

print 'PSF2:', psf2

psf3 = psf2.scaleBy(S)
psf3.radius *= S

print 'PSF3:', psf3

p = psf3.getPointSourcePatch(0., 0.)
print 'psf3 patch', p

plt.clf()
ext = p.getExtent()
ext = [x-0.5 for x in ext]
plt.imshow(p.patch, extent=ext, **ima)
plt.axis([x * S for x in ax0])
plt.colorbar()
ps.savefig()

plt.clf()
plt.imshow(np.log10(np.maximum(p.patch, 1e-10)), vmin=-10,
		   extent=ext, **ima)
plt.axis([x * S for x in ax0])
plt.colorbar()
ps.savefig()

tanwcs = Tan(wcs1.wcs)
tanwcs.scale(S)
wcs3 = FitsWcs(tanwcs)
im1.wcs = wcs3
im1.psf = psf3

mod = tractor.getModelImage(0)
plog(mod)


H,W = H/S, W/S
print 'Pix size', W
data = np.zeros((H,W))
invvar = np.ones((H,W))
pixscale = re * 8. * nre / float(W) / 3600.
print 'Pixscale', pixscale
wcs2 = FitsWcs(Tan(ra, dec, W/2. + 0.5, H/2. + 0.5,
				   pixscale, 0., 0., pixscale, W, H))

im2 = Image(data=data, invvar=invvar, psf=psf2, wcs=wcs2,
			sky=ConstantSky(0.), photocal=NullPhotoCal(),
			name='im2')
tractor = Tractor([im2], [g1])
mod = tractor.getModelImage(0)
plog(mod)
