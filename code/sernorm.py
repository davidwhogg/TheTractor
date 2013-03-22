# Just checking out Jim Bosch's formula for the half-light radius scaling
# "sernorm".
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.plotutils import *

from scipy.special import gammaincinv

def sernorm(n):
	return gammaincinv(2.*n, 0.5)

ps = PlotSequence('sernorm')
X = np.linspace(0., 8., 8001)
dx = (X[1]-X[0])
X = X[:-1] + dx/2.
for n in np.arange(1, 6, 0.5):
	print 'sernorm(%f)' % n, sernorm(n)
	S = np.exp(-sernorm(n) * (X**(1./n) - 1.))
	plt.clf()
	plt.plot(X, S, 'r-')
	cs = np.cumsum(S*X)*dx
	plt.plot(X, cs / (cs[-1]), 'r:')
	plt.axhline(1., color='k', alpha=0.5)
	plt.axhline(0.5, color='k', alpha=0.5)
	plt.axvline(1., color='k', alpha=0.5)
	plt.title('n = %f' % n)
	#ps.savefig()
	plt.xscale('log')
	plt.yscale('log')
	ps.savefig()

