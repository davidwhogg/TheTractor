'''
This file is part of the Tractor project.
Copyright 2011 David W. Hogg.

### bugs:
- When using chi-squared-like badness, we should use levmar not bfgs!
- Spews enormous numbers of warnings when log10_squared_deviation gets very negative.
'''

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
rc('text', usetex=True)
import pylab as plt
import numpy as np
import scipy.optimize as op
from multiprocessing import Pool
import cPickle as pickle

# note wacky normalization because this is for 2-d Gaussians
# (but only ever called in 1-d).  Wacky!
def not_normal(x, V):
    exparg = -0.5 * x * x / V
    result = np.zeros_like(x)
    I = ((exparg > -1000) * (exparg < 1000))
    result[I] = 1. / (2. * np.pi * V) * np.exp(exparg[I])
    return result

# magic number from Ciotti & Bertin, A&A, 352, 447 (1999)
# normalized to return 1. at x=1.
def hogg_exp(x):
    return np.exp(-1.67834699 * (x - 1.))

# magic number from Ciotti & Bertin, A&A, 352, 447 (1999)
# normalized to return 1. at x=1.
def hogg_dev(x):
    return np.exp(-7.66924944 * ((x * x)**0.125 - 1.))

# magic numbers from Lupton (makeprof.c and phFitobj.h) via dstn
# normalized to return 1. at x=1.
def hogg_lup(x):
    inner = 7.
    outer = 8.
    lup = np.exp(-7.66925 * ((x * x + 0.0004)**0.125 - 1.))
    outside = (x >= outer)
    lup[outside] *= 0.
    middle = (x >= inner) * (x <= outer)
    lup[middle] *= (outer - x[middle]) / (outer - inner)
    return lup

# magic numbers from Ciotti & Bertin, A&A, 352, 447 (1999)
# normalized to return 1. at x=1.
def hogg_ser2(x):
    return np.exp(-3.67206075 * ((x * x)**0.25 - 1.))
def hogg_ser3(x):
    return np.exp(-5.67016119 * ((x * x)**(1./6.) - 1.))
def hogg_ser5(x):
    return np.exp(-9.66871461 * ((x * x)**0.1 - 1.))

def hogg_model(x, model):
    if model == 'exp':
        return hogg_exp(x)
    if model == 'dev':
        return hogg_dev(x)
    if model == 'lup':
        return hogg_lup(x)
    if model == 'ser2':
        return hogg_ser2(x)
    if model == 'ser3':
        return hogg_ser3(x)
    if model == 'ser5':
        return hogg_ser5(x)
    assert(1 == 0)
    return None

def mixture_of_not_normals(x, pars):
    K = len(pars)/2
    y = 0.
    for k in range(K):
        y += pars[k] * not_normal(x, pars[k + K])
    return y

def badness_of_fit(lnpars, model, max_radius, log10_squared_deviation):
    """
    `badness_of_fit()`:

    Compute chi-squared deviation between profile and
    mixture-of-Gaussian approximation, where the chi-squared is
    appropriate for two-dimensional intensity fitting.

    Note magic number in penalty for large variance.
    """
    pars = np.exp(lnpars)
    x = np.arange(0.0005, max_radius, 0.001)
    residual = mixture_of_not_normals(x, pars) - hogg_model(x, model)
    # note radius (x) weighting, this makes the differential proportional to 2 * pi * r * dr
    numerator = np.sum(x * residual * residual)
    denominator = np.sum(x)
    badness = numerator / denominator / 10.**log10_squared_deviation
    K = len(pars) / 2
    var = pars[K:]
    extrabadness = 0.0001 * np.sum(var) / max_radius**2
    return badness + extrabadness

def negative_score(lnpars, model, max_radius, log10_squared_deviation):
    """
    `negative_score()`:

    Compute the KL-related score suggested by Brewer.  Plus a penalty
    for amplitude craziness.

    Note magic number in penalty for amplitude at x=1
    """
    pars = np.exp(lnpars)
    extrabadness = 1. * (mixture_of_not_normals(np.ones(1), pars)[0] - 1.)**2
    K = pars.size / 2
    # normalize (REQUIRED):
    pars[0:K] /= np.sum(pars[0:K])
    dx = 0.001
    x = np.arange(0.5 * dx, max_radius, dx)
    # note radius (x) weighting, this makes the differential proportional to 2 * pi * r * dr
    score = np.sum(x * dx * hogg_model(x, model) * np.log(mixture_of_not_normals(x, pars)))
    return extrabadness - score

def optimize_mixture(K, pars, model, max_radius, log10_squared_deviation, badness_fn):
    newlnpars = op.fmin_bfgs(badness_fn, np.log(pars), args=(model, max_radius, log10_squared_deviation))
    return (badness_fn(newlnpars, model, max_radius, log10_squared_deviation), np.exp(newlnpars))

def plot_mixture(pars, prefix, model, max_radius, log10_squared_deviation, badness_fn):
    x2 = np.arange(0.0005, np.sqrt(5. * max_radius), 0.001)**2 # note non-linear spacing
    y1 = hogg_model(x2, model)
    badness = badness_fn(np.log(pars), model, max_radius, log10_squared_deviation)
    K = len(pars) / 2
    y2 = mixture_of_not_normals(x2, pars)
    plt.clf()
    plt.plot(x2, y1, 'k-')
    plt.plot(x2, y2, 'k-', lw=6, alpha=0.25)
    for k in range(K):
        plt.plot(x2, pars[k] * not_normal(x2, pars[k + K]), 'k-', alpha=0.5)
    plt.axvline(max_radius, color='k', alpha=0.25)
    badname = "badness"
    title = r"%s / $K=%d$ / max radius = $%.1f$ / %s = $%.3f\times 10^{%d}$" % (
        model, len(pars) / 2, max_radius, badname, badness, log10_squared_deviation)
    plt.title(title)
    plt.xlim(0., 1.2 * max_radius)
    xlim1 = plt.xlim()
    plt.ylim(-0.1 * 8.0, 1.1 * 8.0)
    xlabel = r"dimensionless angular radius $\xi$"
    plt.xlabel(xlabel)
    plt.ylabel(r"intensity (relative to intensity at $\xi = 1$)")
    plt.savefig(prefix + '_profile.png')
    plt.loglog()
    xmin = 0.001
    yint = np.interp([xmin, ], x2, y1)[0]
    plt.xlim(xmin, 5. * max_radius)
    xlim2 = plt.xlim()
    plt.ylim(1.5e-6 * yint, 1.5 * yint)
    plt.savefig(prefix + '_profile_log.png')
    plt.clf()
    plt.plot(x2, y1 - y2, 'k-')
    plt.plot(x2, 0. * x2, 'k-', lw=6, alpha=0.25)
    plt.axvline(max_radius, color='k', alpha=0.25)
    plt.title(title)
    plt.xlim(*xlim1)
    plt.ylim(-1., 1.)
    plt.xlabel(xlabel)
    plt.ylabel(r"intensity residual (relative to intensity at $\xi = 1$)")
    plt.savefig(prefix + '_residual.png')
    plt.semilogx()
    xmin = 0.001
    plt.xlim(*xlim2)
    plt.savefig(prefix + '_residual_log.png')
    plt.clf()
    plt.plot(x2, (y1 - y2) / y1, 'k-')
    plt.plot(x2, 0. * x2, 'k-', lw=6, alpha=0.25)
    plt.axvline(max_radius, color='k', alpha=0.25)
    plt.title(title)
    plt.xlim(*xlim1)
    plt.ylim(-1., 1.)
    plt.xlabel(xlabel)
    plt.ylabel(r"fractional intensity residual")
    plt.savefig(prefix + '_fractional.png')
    plt.semilogx()
    xmin = 0.001
    plt.xlim(*xlim2)
    plt.savefig(prefix + '_fractional_log.png')
    return None

def rearrange_pars(pars):
    K = len(pars) / 2
    indx = np.argsort(pars[K: K + K])
    amp = pars[indx]
    var = pars[K+indx]
    return np.append(amp, var)

# run this (possibly with adjustments to the magic numbers at top)
# to find different or better mixtures approximations
def main(input):
    model = input
    max_radius = 8.0
    log10_squared_deviation = -2.
    amp = np.array([1.0])
    var = np.array([1.0])
    pars = np.append(amp, var)
    if True:
        bad_fn = badness_of_fit
    else:
        bad_fn = negative_score
    (badness, pars) = optimize_mixture(1, pars, model, max_radius, log10_squared_deviation, bad_fn)
    lastKbadness = badness
    bestbadness = badness
    for K in range(2, 20):
        prefix = '%s_K%02d_MR%02d' % (model, K, int(round(max_radius) + 0.01))
        print 'working on %s at K = %d (%s)' % (model, K, prefix)
        newvar = 2.0 * np.max(np.append(var, 1.0))
        newamp = 1.0 * newvar
        amp = np.append(newamp, amp)
        var = np.append(newvar, var)
        pars = np.append(amp, var)
        for i in range(2 * K):
            (badness, pars) = optimize_mixture(K, pars, model, max_radius, log10_squared_deviation, bad_fn)
            if (badness < bestbadness) or (i == 0):
                print '%s %d %d improved' % (model, K, i)
                bestpars = rearrange_pars(pars)
                bestbadness = badness
                while bestbadness < 1. and log10_squared_deviation > -5.5:
                    bestbadness *= 10.
                    log10_squared_deviation = np.round(log10_squared_deviation - 1.)
            else:
                print '%s %d %d not improved' % (model, K, i)
                amp = 1. * bestpars[0:K]
                var = 1. * bestpars[K:K+K]
                var[0] = 2.0 * var[np.mod(i, K)]
                amp[0] = 0.5 * amp[np.mod(i, K)]
                pars = np.append(amp, var)
        lastKbadness = bestbadness
        pars = rearrange_pars(bestpars)
        amp = pars[0:K]
        var = pars[K:K+K]
        plot_mixture(pars, prefix, model, max_radius, log10_squared_deviation, bad_fn)
        txtfile = open(prefix + '.txt', "w")
        txtfile.write("%s\n" % repr(pars))
        txtfile.close()
        picklefile = open(prefix + '.pickle', "wb")
        pickle.dump(pars, picklefile)
        picklefile.close()
        if bestbadness < 10. and log10_squared_deviation < -5.5 and K > 11:
            break
    return None

if __name__ == '__main__':
    if False: # use multiprocessing
        pmap = Pool(6).map
    else: # don't use multiprocessing
        pmap = map
    inputs = [
        'exp',
        'dev',
        'lup',
        'ser2',
        'ser3',
        'ser5',
        ]
    inputs = ['lup', ]
    pmap(main, inputs)
