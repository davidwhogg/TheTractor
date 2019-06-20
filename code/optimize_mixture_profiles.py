'''
This file is part of the Tractor project.
Copyright 2011, 2012 David W. Hogg.

### bugs:
- When using chi-squared-like badness, we should use levmar not bfgs!
- Spews enormous numbers of warnings when log10_squared_deviation gets very negative.
'''
from __future__ import print_function
try:
    # py2
    import cPickle as pickle
except:
    import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','size':12})
rc('text', usetex=True)
import pylab as plt
import numpy as np
import scipy.optimize as op
from multiprocessing import Pool
import os as os
from sernorm import sernorm

def not_normal(x, V):
    """
    Make a one-dimensional profile of a two-dimensional Gaussian.

    Note wacky normalization because this is for 2-d Gaussians
    (but only ever called in 1-d).  Wacky!
    """
    exparg = -0.5 * x * x / V
    result = np.zeros_like(x)
    I = ((exparg > -1000) * (exparg < 1000))
    result[I] = 1. / (2. * np.pi * V) * np.exp(exparg[I])
    return result

def hogg_ser(x, n, soft=None):
    if soft:
        return np.exp(-sernorm(n) * ((x**2 + soft) ** (1./(2.*n)) - 1.))
    return np.exp(-sernorm(n) * (x ** (1./n) - 1.))

def hogg_exp(x):
    """
    One-dimensional exponential profile.

    Normalized to return 1. at x=1.
    """
    return hogg_ser(x, 1.)

def hogg_dev(x):
    """
    One-dimensional DeVaucouleurs profile.

    normalized to return 1. at x=1.
    """
    return hogg_ser(x, 4.)

def hogg_luv(x):
    """
    One-dimensional Lupton approximation to DeVaucouleurs profile.

    Normalized to return 1. at x=1.
    """
    inner = 7.
    outer = 8.
    luv = hogg_ser(x, 4., soft=4e-4)
    luv[x > outer] *= 0.
    middle = (x >= inner) * (x <= outer)
    luv[middle] *= (1. - ((x[middle] - inner) / (outer - inner)) ** 2) ** 2
    return luv

def hogg_lux(x):
    """
    One-dimensional Lupton approximation to exponential profile.

    Magic numbers from Lupton (makeprof.c and phFitobj.h) via dstn.
    Normalized to return 1. at x=1.
    """
    inner = 3.
    outer = 4.
    lux = hogg_ser(x, 1.)
    lux[x > outer] *= 0.
    middle = (x >= inner) * (x <= outer)
    lux[middle] *= (1. - ((x[middle] - inner) / (outer - inner)) ** 2) ** 2
    return lux

def hogg_ser2(x):
    """
    One-dimensional S\'ersic profile at n=2.

    Normalized to return 1. at x=1.
    """
    return hogg_ser(x, 2.)

def hogg_ser3(x):
    """
    One-dimensional S\'ersic profile at n=3.

    Normalized to return 1. at x=1.
    """
    return hogg_ser(x, 3.)

def hogg_ser5(x):
    """
    One-dimensional S\'ersic profile at n=5.

    Normalized to return 1. at x=1.
    """
    return hogg_ser(x, 5.)

def hogg_model(x, model):
    """
    Interface to all one-dimensional profiles.
    """
    if model == 'exp':
        return hogg_exp(x)
    elif model == 'dev':
        return hogg_dev(x)
    elif model == 'luv':
        return hogg_luv(x)
    elif model == 'lux':
        return hogg_lux(x)
    elif model == 'ser2':
        return hogg_ser2(x)
    elif model == 'ser3':
        return hogg_ser3(x)
    elif model == 'ser5':
        return hogg_ser5(x)
    return hogg_ser(x, model)

def mixture_of_not_normals(x, pars):
    """
    One-dimensional profile made from a mixture of two dimensional
    concentric Gaussians.
    """
    K = len(pars)//2
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
    
    Magic number in extrabadness.
    """
    pars = np.exp(lnpars)
    x = np.arange(0.0005, max_radius, 0.001)
    residual = mixture_of_not_normals(x, pars) - hogg_model(x, model)
    # note radius (x) weighting, this makes the differential proportional to 2 * pi * r * dr
    numerator = np.sum(x * residual * residual)
    denominator = np.sum(x)
    badness = numerator / denominator
    K = len(pars) // 2
    var = pars[K:]
    extrabadness = 1e-10 * np.mean(var / max_radius**2)
    if extrabadness > badness:
        print("EXTRABAD:", model, pars, max_radius, log10_squared_deviation)
    return (badness + extrabadness) / 10.**log10_squared_deviation

def negative_score(lnpars, model, max_radius, log10_squared_deviation):
    """
    `negative_score()`:

    Compute the KL-related score suggested by Brewer.  Plus a penalty
    for amplitude craziness.

    Note magic number in penalty for amplitude at x=1
    """
    pars = np.exp(lnpars)
    extrabadness = 1. * (mixture_of_not_normals(np.ones(1), pars)[0] - 1.)**2
    K = pars.size // 2
    # normalize (REQUIRED):
    pars[0:K] /= np.sum(pars[0:K])
    dx = 0.001
    x = np.arange(0.5 * dx, max_radius, dx)
    # note radius (x) weighting, this makes the differential proportional to 2 * pi * r * dr
    score = np.sum(x * dx * hogg_model(x, model) * np.log(mixture_of_not_normals(x, pars)))
    return extrabadness - score

def optimize_mixture(K, pars, model, max_radius, log10_squared_deviation, badness_fn):
    lnpars = np.log(pars)
    newlnpars = op.fmin_powell(badness_fn, lnpars, args=(model, max_radius, log10_squared_deviation), maxfun=16384 * 2)
    lnpars = 1. * newlnpars
    newlnpars = op.fmin_bfgs(badness_fn, lnpars, args=(model, max_radius, log10_squared_deviation), maxiter=128 * 2)
    lnpars = 1. * newlnpars
    newlnpars = op.fmin_cg(badness_fn, lnpars, args=(model, max_radius, log10_squared_deviation), maxiter=128 * 2)
    return (badness_fn(newlnpars, model, max_radius, log10_squared_deviation), np.exp(newlnpars))

def hogg_savefig(prefix):
    fn = prefix + '.png'
    print("writing %s" % fn)
    plt.savefig(fn)
    fn = prefix + '.pdf'
    print("writing %s" % fn)
    plt.savefig(fn)

def plot_mixture(pars, prefix, model, max_radius, log10_squared_deviation, badness_fn):
    x2 = np.arange(0.0005, np.sqrt(5. * max_radius), 0.001)**2 # note non-linear spacing
    y1 = hogg_model(x2, model)
    badness = badness_fn(np.log(pars), model, max_radius, log10_squared_deviation)
    K = len(pars) // 2
    y2 = mixture_of_not_normals(x2, pars)
    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(x2, y1, 'k-')
    plt.plot(x2, y2, 'k-', lw=6, alpha=0.25)
    for k in range(K):
        plt.plot(x2, pars[k] * not_normal(x2, pars[k + K]), 'k-', alpha=0.5)
    plt.axvline(max_radius, color='k', alpha=0.25)
    badname = "badness"
    title = r"%s / $M^{\mathrm{%s}}=%d$ / $\xi_{\max} = %d$ / %s $= %.2f\times 10^{%d}$" % (
        model, model, len(pars) / 2, max_radius, badname, badness, log10_squared_deviation)
    plt.title(title)
    plt.xlim(-0.5, 8.5)
    xlim1 = plt.xlim()
    plt.ylim(-0.1 * 8.0, 1.1 * 8.0)
    ylim1 = plt.ylim()
    xlabel = r"dimensionless angular radius $\xi$"
    plt.xlabel(xlabel)
    plt.ylabel(r"intensity (relative to intensity at $\xi = 1$)")
    hogg_savefig(prefix + '_profile')
    plt.semilogy()
    xmin = 0.001
    yint = np.interp([xmin, ], x2, y1)[0]
    plt.xlim(xlim1)
    plt.ylim(1.5e-6 * yint, 1.5 * yint)
    ylim2 = plt.ylim()
    hogg_savefig(prefix + '_profile_semilog')
    plt.loglog()
    plt.xlim(xmin, 12.)
    xlim2 = plt.xlim()
    plt.ylim(ylim2)
    hogg_savefig(prefix + '_profile_log')
    plt.clf()
    plt.plot(x2, y1 - y2, 'k-')
    plt.plot(x2, 0. * x2, 'k-', lw=6, alpha=0.25)
    plt.axvline(max_radius, color='k', alpha=0.25)
    plt.title(title)
    plt.xlim(*xlim1)
    plt.ylim(-1., 1.)
    plt.xlabel(xlabel)
    plt.ylabel(r"intensity residual (units of intensity at $\xi = 1$)")
    hogg_savefig(prefix + '_residual')
    plt.semilogx()
    xmin = 0.001
    plt.xlim(*xlim2)
    hogg_savefig(prefix + '_residual_log')
    plt.clf()
    plt.plot(x2, (y1 - y2) / y1, 'k-')
    plt.plot(x2, 0. * x2, 'k-', lw=6, alpha=0.25)
    plt.axvline(max_radius, color='k', alpha=0.25)
    plt.title(title)
    plt.xlim(*xlim1)
    plt.ylim(-1., 1.)
    plt.xlabel(xlabel)
    plt.ylabel(r"fractional intensity residual (residual / intensity)")
    hogg_savefig(prefix + '_fractional')
    plt.semilogx()
    xmin = 0.001
    plt.xlim(*xlim2)
    hogg_savefig(prefix + '_fractional_log')
    return None

def rearrange_pars(pars):
    K = len(pars) // 2
    indx = np.argsort(pars[K: K + K])
    amp = pars[indx]
    var = pars[K+indx]
    return np.append(amp, var)

# run this (possibly with adjustments to the magic numbers at top)
# to find different or better mixtures approximations
def main(input):
    model, max_radius = input
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
        picklefn = prefix + '.pickle'
        if os.path.exists(picklefn):
            print('not working on %s at K = %d (%s); just reading %s' % (model, K, prefix, picklefn))
            picklefile = open(picklefn, "r")
            pars = pickle.load(picklefile)
            picklefile.close()
            bestbadness = bad_fn(np.log(pars), model, max_radius, log10_squared_deviation)
            while bestbadness < 1. and log10_squared_deviation > -7.5:
                bestbadness *= 10.
                log10_squared_deviation = np.round(log10_squared_deviation - 1.)
                print("K, bestbadness, log10_squared_deviation", K, bestbadness, log10_squared_deviation)
        else:
            print('working on %s at K = %d (%s)' % (model, K, prefix))
            newvar = 2.0 * np.max(np.append(var, 1.0))
            newamp = 1.0 * newvar
            amp = np.append(newamp, amp)
            var = np.append(newvar, var)
            pars = np.append(amp, var)
            for i in range(2 * K):
                (badness, pars) = optimize_mixture(K, pars, model, max_radius, log10_squared_deviation, bad_fn)
                if (badness < (bestbadness - 1.e-7)) or (i == 0):
                    print('%s %d %d improved' % (model, K, i))
                    bestpars = rearrange_pars(pars)
                    bestbadness = badness
                    while bestbadness < 1. and log10_squared_deviation > -7.5:
                        bestbadness *= 10.
                        log10_squared_deviation = np.round(log10_squared_deviation - 1.)
                        print("K, bestbadness, log10_squared_deviation", K, bestbadness, log10_squared_deviation)
                else:
                    print('%s %d %d not improved' % (model, K, i))
                    amp = 1. * bestpars[0:K]
                    var = 1. * bestpars[K:K+K]
                    var[0] = 2.0 * var[np.mod(i, K)]
                    amp[0] = 0.5 * amp[np.mod(i, K)]
                    pars = np.append(amp, var)
            lastKbadness = bestbadness
            pars = rearrange_pars(bestpars)
            picklefile = open(picklefn, "wb")
            pickle.dump(pars, picklefile)
            picklefile.close()
        amp = pars[0:K]
        var = pars[K:K+K]
        plot_mixture(pars, prefix, model, max_radius, log10_squared_deviation, bad_fn)
        txtfile = open(prefix + '.txt', "w")
        txtfile.write("%s\n" % repr(pars))
        txtfile.close()
        if K > 11:
            break
    return None

if __name__ == '__main__':
    main((0.4, 8.))
    sys.exit(0)
    
    if True: # use multiprocessing
        pmap = Pool(8).map
    else: # don't use multiprocessing
        pmap = map
    # inputs = [
    #     ('dev', 8.),
    #     ('luv', 8.),
    #     ('ser2', 8.),
    #     ('ser3', 8.),
    #     ('ser5', 8.),
    #     ('exp', 8.),
    #     ('lux', 4.),
    #     ]
    #inputs = [(x, 8.) for x in np.arange(0.5, 6)]
    inputs = [(x, 8.) for x in np.arange(0.75, 6, 0.5)]
    pmap(main, inputs)

