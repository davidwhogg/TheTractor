# Just checking out Jim Bosch's formula for the half-light radius scaling
# "sernorm".
from scipy.special import gammaincinv

def sernorm(n):
	return gammaincinv(2.*n, 0.5)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt
    import numpy as np
    from astrometry.util.plotutils import *
    ps = PlotSequence('sernorm')
    X = np.linspace(0., 8., 8001)
    dx = (X[1]-X[0])
    X = X[:-1] + dx/2.
    cc = ['k', 'r','b','g','m','y','c','k',(1.,0.5,0),'0.3',(0,0.5,1.)]
    plt.clf()
    for i,n in enumerate(np.arange(0.5, 6, 0.5)):
    	print 'sernorm(%f)' % n, sernorm(n)
    	S = np.exp(-sernorm(n) * (X**(1./n) - 1.))
    	plt.plot(X, S, '-', color=cc[i])
    	cs = np.cumsum(S*X)*dx
    	plt.plot(X, cs / (cs[-1]), ':', color=cc[i])
    plt.axhline(1., color='k', alpha=0.5)
    plt.axhline(0.5, color='k', alpha=0.5)
    plt.axvline(1., color='k', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(X.min(), X.max())
    ps.savefig()
