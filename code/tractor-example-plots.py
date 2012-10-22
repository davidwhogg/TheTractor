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

ps = PlotSequence('trdemo')

ra,dec = 0.,0.
rd = RaDecPos(ra, dec)
m1 = Flux(100.)
re = 1.
s1 = GalaxyShape(re, 1., 0.)
s2 = GalaxyShape(re, 0.5, 40.)
g1 = DevGalaxy(rd, m1, s1)

H,W = 500,500
data = np.zeros((H,W))
invvar = np.ones((H,W))
psf1 = NCircularGaussianPSF([1e-3], [1.])
pixscale = re * 8. * 2. / float(W) / 3600.
wcs1 = FitsWcs(Tan(ra, dec, W/2. + 0.5, H/2. + 0.5,
				   pixscale, 0., 0., pixscale, W, H))

im1 = Image(data=data, invvar=invvar, psf=psf1, wcs=wcs1,
			sky=ConstantSky(0.), photocal=NullPhotoCal(),
			name='im1')

tractor = Tractor([im1], [g1])

mod = tractor.getModelImage(0)

ima = dict(interpolation='nearest', origin='lower')

logmrange = np.max(np.log(mod)) - np.min(np.log(mod))

plt.clf()
plt.imshow(mod, **ima)
plt.gray()
plt.colorbar()
ax = plt.axis()
ps.savefig()

plt.clf()
plt.imshow(np.log(mod), **ima)
plt.gray()
plt.colorbar()
ps.savefig()

angles = np.linspace(0., 2.*np.pi, 360, endpoint=False)
#i0 = 0
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
S = 4.
dradec *= S
dra  = dradec[:,0]
ddec = dradec[:,1]
xy = np.array([wcs1.positionToPixel(RaDecPos(ra + idra, dec + iddec))
			   for idra,iddec in zip(dra,ddec)])
xx,yy = xy[:,0], xy[:,1]

plt.clf()
plt.plot(xx, yy, 'k-')
plt.axis(ax)
ps.savefig()

S = 3600.
plt.clf()
plt.plot(dra * S, ddec * S, 'k-')
for i in [i0, i90]:
	#print 'i', i
	L = 0.3
	plt.arrow(0, 0, dra[i]*S, ddec[i]*S, shape='full',
			  overhang=0, head_length=L, head_width=0.5*L, fc='k',
			  length_includes_head=True)
plt.axvline(0., color='k', alpha=0.3)
plt.axhline(0., color='k', alpha=0.3)
plt.axis('equal')
ps.savefig()

mod0 = mod

g1.shape = s2
#g1.shape.setParams(s2.getParams())
mod = tractor.getModelImage(0)

plt.clf()
plt.imshow(np.log(mod), vmax=0., vmin=np.min(np.log(mod0)), **ima)
plt.gray()
plt.colorbar()
ps.savefig()


#psf2 = GaussianMixturePSF(np.array([0.5, 0.25, 0.25]),
#						  np.array([[0.,0.], [1.,0.], [0,-1]]),
#						  np.array([[[1.,0.,],[0.,1.]],
#									[[0.1,0.8],[0.8,0.3]],
#									[[0.4,2.0],[2.0,0.4]]]))

t = np.deg2rad(70.)
R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
o = np.array([[2.,0.],[0.,0.5]])
V2 = np.dot(R.T, np.dot(o, R))
print 'V', V2

V1 = np.eye(2) * 3.
print 'V', V1

#t = np.deg2rad(-30.)
#R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
#o = np.array([[2.,0.],[0.,0.25]])
#V3 = np.dot(R.T, np.dot(o, R))
#V3 = np.eye(2) * 4.
V3 = np.array([[4., -0.5],[-0.5,3.]])

VV = np.array([V1, V2, V3])
print 'VV', VV

# psf2 = GaussianMixturePSF(np.array([0.5, 0.25, 0.25]),
# 						  np.array([[0.,0.], [1.,0.], [0.,-1.]]),
# 						  np.array([[[1.,0.,],[0.,1.]],
# 									[[2.,0.8],[0.8,2.]],
# 									[[0.5,0.],[0., 2.]],
# 									]))
psf2 = GaussianMixturePSF(np.array([0.6, 0.6, 0.6]),
						  np.array([[0.,0.], [-.5,0.], [1.,-0.5]]), VV)
im1.psf = psf2

p = psf2.getPointSourcePatch(25., 25.)

plt.clf()
plt.imshow(p.patch, **ima)
plt.gray()
plt.colorbar()
ps.savefig()

plt.clf()
plt.imshow(np.log(np.maximum(p.patch, 1e-20)), vmin=-20, **ima)
plt.gray()
plt.colorbar()
ps.savefig()

S = 5

# psf2.scaleBy(5.) ?
psf3 = psf2.scaleBy(S)
psf3.radius *= S

p = psf3.getPointSourcePatch(25., 25.)

plt.clf()
plt.imshow(p.patch, **ima)
plt.gray()
plt.colorbar()
ps.savefig()

plt.clf()
plt.imshow(np.log(np.maximum(p.patch, 1e-20)), vmin=-20, **ima)
plt.gray()
plt.colorbar()
ps.savefig()

tanwcs = Tan(wcs1.wcs)
tanwcs.scale(S)
wcs2 = FitsWcs(tanwcs)
im1.wcs = wcs2
im1.psf = psf3

mod = tractor.getModelImage(0)
mx = np.log(mod).max()
plt.clf()
plt.imshow(np.log(mod), vmax=mx, vmin=mx-logmrange, **ima)
plt.gray()
plt.colorbar()
ps.savefig()



H,W = H/S, W/S
data = np.zeros((H,W))
invvar = np.ones((H,W))
pixscale = re * 8. * 2. / float(W) / 3600.
wcs4 = FitsWcs(Tan(ra, dec, W/2. + 0.5, H/2. + 0.5,
				   pixscale, 0., 0., pixscale, W, H))

im4 = Image(data=data, invvar=invvar, psf=psf2, wcs=wcs4,
			sky=ConstantSky(0.), photocal=NullPhotoCal(),
			name='im4')
tractor = Tractor([im4], [g1])
mod = tractor.getModelImage(0)

plt.clf()
mx = np.log(mod).max()
plt.imshow(np.log(mod), vmax=mx, vmin=mx-logmrange, **ima)
plt.gray()
plt.colorbar()
ps.savefig()
