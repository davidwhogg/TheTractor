from math import ceil, floor, pi, sqrt, exp

from engine import *
from ducks import *
from utils import *
import mixture_profiles as mp


class Mag(ScalarParam):
	'''
	An implementation of Brightness that stores a single Mag.
	'''
	stepsize = 0.01
	strformat = '%.3f'

class Flux(ScalarParam):
	'''
	A simple one-band Flux implementation of Brightness.
	'''
	def __mul__(self, factor):
		new = self.copy()
		new.val *= factor
		return new
	__rmul__ = __mul__
	# enforce limit: Flux > 0
	def _set(self, val):
		if val < 0:
			#print 'Clamping Flux from', p[0], 'to zero'
			pass
		self.val = max(0., val)

class NullPhotoCal(object):
	'''
	The "identity" PhotoCal -- the Brightness objects are in units of Image counts.
	'''
	def brightnessToCounts(self, brightness):
		return brightness.getValue()
	def countsToBrightness(self, counts):
		return counts.getValue()


class NullWCS(WCS):
	'''
	The "identity" WCS -- useful when you are using raw pixel
	positions rather than RA,Decs.
	'''
	def __init__(self, pixscale=1.):
		'''
		pixscale: [arcsec/pix]
		'''
		self.pixscale = pixscale
	def hashkey(self):
		return ('NullWCS',)
	def positionToPixel(self, src, pos):
		return pos
	def pixelToPosition(self, src, xy):
		return xy
	def cdAtPixel(self, x, y):
		return np.array([[1.,0.],[0.,1.]]) * self.pixscale / 3600.

class FitsWcs(object):
	'''
	A WCS implementation that wraps a FITS WCS object (possibly with a
	pixel offset)
	'''
	def __init__(self, wcs):
		self.wcs = wcs
		self.x0 = 0
		self.y0 = 0

	def hashkey(self):
		return ('FitsWcs', self.x0, self.y0, self.wcs)

	def setX0Y0(self, x0, y0):
		self.x0 = x0
		self.y0 = y0

	def positionToPixel(self, src, pos):
		#ok,x,y = self.wcs.radec2pixelxy(pos.ra, pos.dec)
		X = self.wcs.radec2pixelxy(pos.ra, pos.dec)
		if len(X) == 3:
			ok,x,y = X
		else:
			assert(len(X) == 2)
			x,y = X
		return x-self.x0, y-self.y0

	def pixelToPosition(self, src, xy):
		(x,y) = xy
		r,d = self.wcs.pixelxy2radec(x + self.x0, y + self.y0)
		return RaDecPos(r,d)

	def cdAtPixel(self, x, y):
		cd = self.wcs.get_cd()
		return np.array([[cd[0], cd[1]], [cd[2],cd[3]]])


class PixPos(ParamList):
	'''
	A Position implementation using pixel positions.
	'''
	def getNamedParams(self):
		return [('x', 0), ('y', 1)]
	def __str__(self):
		return 'pixel (%.2f, %.2f)' % (self.x, self.y)
	#def __repr__(self):
	#	return 'PixPos(%.4f, %.4f)' % (self.x, self.y)
	#def copy(self):
	#	return PixPos(self.x, self.y)
	#def hashkey(self):
	#	return ('PixPos', self.x, self.y)
	def getDimension(self):
		return 2
	def getStepSizes(self, img, *args, **kwargs):
		return [0.1, 0.1]

class RaDecPos(ParamList):
	'''
	A Position implementation using RA,Dec positions.
	'''
	def getNamedParams(self):
		return [('ra', 0), ('dec', 1)]
	def __str__(self):
		return 'RA,Dec (%.5f, %.5f)' % (self.ra, self.dec)
	#def __repr__(self):
	#	return 'RaDecPos(%.5f, %.5f)' % (self.ra, self.dec)
	#def copy(self):
	#	return RaDecPos(self.ra, self.dec)
	#def hashkey(self):
	#	return ('RaDecPos', self.ra, self.dec)
	def getDimension(self):
		return 2
	def getStepSizes(self, img, *args, **kwargs):
		return [1e-4, 1e-4]

class ConstantSky(ScalarParam):
	'''
	In counts
	'''
	def getParamDerivatives(self, img, brightnessonly=False):
		p = Patch(0, 0, np.ones(img.shape))
		p.setName('dsky')
		return [p]
	def addTo(self, img):
		img += self.val
	

class PointSource(MultiParams):
	'''
	An implementation of a point source, characterized by its position
	and brightness.
	'''
	def __init__(self, pos, brightness):
		MultiParams.__init__(self, pos, brightness)
	def getSourceType(self):
		return 'PointSource'
	def getNamedParams(self):
		return [('pos', 0), ('brightness', 1)]
	def getPosition(self):
		return self.pos
	def getBrightness(self):
		return self.brightness
	def setBrightness(self, brightness):
		self.brightness = brightness
	def __str__(self):
		return (self.getSourceType() + ' at ' + str(self.pos) +
				' with ' + str(self.brightness))
	def __repr__(self):
		return (self.getSourceType() + '(' + repr(self.pos) + ', ' +
				repr(self.brightness) + ')')
	#def copy(self):
	#	return PointSource(self.pos.copy(), self.brightness.copy())
	#def hashkey(self):
	#	return ('PointSource', self.pos.hashkey(), self.brightness.hashkey())

	def getModelPatch(self, img):
		(px,py) = img.getWcs().positionToPixel(self, self.getPosition())
		patch = img.getPsf().getPointSourcePatch(px, py)
		counts = img.getPhotoCal().brightnessToCounts(self.brightness)
		return patch * counts

	def getParamDerivatives(self, img, brightnessonly=False):
		'''
		returns [ Patch, Patch, ... ] of length numberOfParams().
		'''
		pos0 = self.getPosition()
		(px0,py0) = img.getWcs().positionToPixel(self, pos0)
		patch0 = img.getPsf().getPointSourcePatch(px0, py0)
		counts0 = img.getPhotoCal().brightnessToCounts(self.brightness)
		derivs = []
		psteps = pos0.getStepSizes(img)
		if brightnessonly or self.isParamPinned('pos'):
			derivs.extend([None] * len(psteps))
		else:
			pvals = pos0.getParams()
			print 'position params:', pvals
			print 'position steps:', psteps
			for i,pstep in enumerate(psteps):
				oldval = pos0.setParam(i, pvals[i] + pstep)
				(px,py) = img.getWcs().positionToPixel(self, pos0)
				patchx = img.getPsf().getPointSourcePatch(px, py)
				pos0.setParam(i, oldval)
				dx = (patchx - patch0) * (counts0 / pstep)
				dx.setName('d(ptsrc)/d(pos%i)' % i)
				derivs.append(dx)

		bsteps = self.brightness.getStepSizes(img)
		if self.isParamPinned('brightness'):
			derivs.extend([None] * len(bsteps))
		else:
			bvals = self.brightness.getParams()
			for i,bstep in enumerate(bsteps):
				oldval = self.brightness.setParam(i, bvals[i] + bstep)
				countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
				self.brightness.setParam(i, oldval)
				df = patch0 * ((countsi - counts0) / bstep)
				df.setName('d(ptsrc)/d(bright%i)' % i)
				derivs.append(df)
		return derivs


class GaussianMixturePSF(Params):
	'''
	A PSF model that is a mixture of general 2-D Gaussians
	(characterized by amplitude, mean, covariance)
	'''
	# Call into MOG to set params, or keep my own copy (via MultiParams)
	def __init__(self, amp, mean, var):
		self.mog = mp.MixtureOfGaussians(amp, mean, var)
		assert(self.mog.D == 2)
		self.radius = 25

	def getMixtureOfGaussians(self):
		return self.mog
	def proposeIncreasedComplexity(self, img):
		raise
	def getStepSizes(self, img, *args, **kwargs):
		K = self.mog.K
		# amp + mean + var
		# FIXME: -1 for normalization?
		#  : -K for variance symmetry
		return [0.01]*K + [0.01]*(K*2) + [0.1]*(K*3)

	def isValidParamStep(self, dparam):
		## FIXME
		return True
	def applyTo(self, image):
		raise
	def getNSigma(self):
		# MAGIC -- N sigma for rendering patches
		return 5.
	def getRadius(self):
		# sqrt(det(var)) ?
		# hmm, really, max(eigenvalue)
		# well, enclosing circle of mu + Nsigma * eigs
		#K = self.mog.K
		#return self.getNSigma * np.max(self.mog.
		# HACK!
		return self.radius
	# returns a Patch object.
	def getPointSourcePatch(self, px, py):
		r = self.getRadius()
		x0,x1 = int(floor(px-r)), int(ceil(px+r))
		y0,y1 = int(floor(py-r)), int(ceil(py+r))
		grid = self.mog.evaluate_grid_dstn(x0-px, x1-px, y0-py, y1-py)
		return Patch(x0, y0, grid)

	#def getNamedParams(self):
	#	return [('sigmas', 0), ('weights', 1)]
	def __str__(self):
		return 'GaussianMixturePSF: ' + str(self.mog)
	def hashkey(self):
		return ('GaussianMixturePSF',
				tuple(self.mog.amp.ravel()),
				tuple(self.mog.mean.ravel()),
				tuple(self.mog.var.ravel()),)
	
	def copy(self):
		raise
		#return self.mog.copy()

	def numberOfParams(self):
		K = self.mog.K
		return K * (1 + 2 + 3)

	# def stepParam(self, parami, delta):
	# 	K = self.mog.K
	# 	if parami < K:
	# 		self.mog.amp[parami] += delta
	# 		return
	# 	parami -= K
	# 	if parami < (K*2):
	# 		i,j = parami / 2, parami % 2
	# 		self.mog.mean[i,j] += delta
	# 		return
	# 	parami -= 2*K
	# 	i,j = parami / 3, parami % 3
	# 	if j in [0,1]:
	# 		self.mog.var[i,j,j] += deltai
	# 	else:
	# 		self.mog.var[i,0,1] += deltai
	# 		self.mog.var[i,1,0] += deltai

	# Returns a *copy* of the current parameter values (list)
	def getParams(self):
		p = list(self.mog.amp) + list(self.mog.mean.ravel())
		for v in self.mog.var:
			p += (v[0,0], v[1,1], v[0,1])
		return p

	def setParams(self, p):
		K = self.mog.K
		self.mog.amp = p[:K]
		pp = p[K:]
		self.mog.mean = np.array(pp[:K*2]).reshape(K,D)
		pp = pp[K*2:]
		self.mog.var[:,0,0] = pp[::3]
		self.mog.var[:,1,1] = pp[1::3]
		self.mog.var[:,0,1] = self.mog.var[:,1,0] = pp[2::3]

	
class NCircularGaussianPSF(MultiParams):
	'''
	A PSF model using N concentric, circular Gaussians.
	'''
	def __init__(self, sigmas, weights):
		'''
		Creates a new N-Gaussian (concentric, isotropic) PSF.

		sigmas: (list of floats) standard deviations of the components

		weights: (list of floats) relative weights of the components;
		given two components with weights 0.9 and 0.1, the total mass
		due to the second component will be 0.1.  These values will be
		normalized so that the total mass of the PSF is 1.0.

		eg,   NCircularGaussianPSF([1.5, 4.0], [0.8, 0.2])
		'''
		assert(len(sigmas) == len(weights))
		MultiParams.__init__(self, ParamList(*sigmas), ParamList(*weights))

	def getNamedParams(self):
		return [('sigmas', 0), ('weights', 1)]

	def __str__(self):
		return ('NCircularGaussianPSF: sigmas [ ' +
				', '.join(['%.3f'%s for s in self.sigmas]) +
				' ], weights [ ' +
				', '.join(['%.3f'%w for w in self.weights]) +
				' ]')

	def __repr__(self):
		return ('NCircularGaussianPSF: sigmas [ ' +
				', '.join(['%.3f'%s for s in self.sigmas]) +
				' ], weights [ ' +
				', '.join(['%.3f'%w for w in self.weights]) +
				' ]')

	def getMixtureOfGaussians(self):
		return mp.MixtureOfGaussians(self.weights,
									 np.zeros((len(self.weights), 2)),
									 np.array(self.sigmas)**2)
		
	def proposeIncreasedComplexity(self, img):
		maxs = np.max(self.sigmas)
		# MAGIC -- make new Gaussian with variance bigger than the biggest
		# so far
		return NCircularGaussianPSF(list(self.sigmas) + [maxs + 1.],
							list(self.weights) + [0.05])

	def getStepSizes(self, img, *args, **kwargs):
		N = len(self.sigmas)
		return [0.01]*N + [0.01]*N

	'''
	def isValidParamStep(self, dparam):
		NS = len(self.sigmas)
		assert(len(dparam) == 2*NS)
		dsig = dparam[:NS]
		dw = dparam[NS:]
		for s,ds in zip(self.sigmas, dsig):
			# MAGIC
			if s + ds < 0.1:
				return False
		for w,dw in zip(self.weights, dw):
			if w + dw < 0:
				return False
		return True
		#return all(self.sigmas + dsig > 0.1) and all(self.weights + dw > 0)
		'''

	def normalize(self):
		mx = max(self.weights)
		self.weights.setParams([w/mx for w in self.weights])

	def hashkey(self):
		return ('NCircularGaussianPSF', tuple(self.sigmas), tuple(self.weights))
	
	def copy(self):
		return NCircularGaussianPSF(list([s for s in self.sigmas]),
							list([w for w in self.weights]))

	def applyTo(self, image):
		from scipy.ndimage.filters import gaussian_filter
		# gaussian_filter normalizes the Gaussian; the output has ~ the
		# same sum as the input.
		
		res = np.zeros_like(image)
		for s,w in zip(self.sigmas, self.weights):
			res += w * gaussian_filter(image, s)
		res /= sum(self.weights)
		return res

	def getNSigma(self):
		# HACK - MAGIC -- N sigma for rendering patches
		return 5.

	def getRadius(self):
		return max(self.sigmas) * self.getNSigma()

	# returns a Patch object.
	def getPointSourcePatch(self, px, py):
		ix = int(round(px))
		iy = int(round(py))
		dx = px - ix
		dy = py - iy

		rad = int(ceil(self.getRadius()))
		sz = 2*rad + 1
		X,Y = np.meshgrid(np.arange(sz).astype(float), np.arange(sz).astype(float))
		X -= dx + rad
		Y -= dy + rad
		patch = np.zeros((sz,sz))
		x0 = ix - rad
		y0 = iy - rad
		R2 = (X**2 + Y**2)
		for s,w in zip(self.sigmas, self.weights):
			patch += w / (2.*pi*s**2) * np.exp(-0.5 * R2 / (s**2))
		patch /= sum(self.weights)
		#print 'sum of PSF patch:', patch.sum()
		return Patch(x0, y0, patch)

