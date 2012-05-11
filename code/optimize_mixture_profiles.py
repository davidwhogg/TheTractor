'''
this file is part of the Tractor project.
Copyright 2011 David W. Hogg.

### bugs:
- should use levmar not bfgs!
'''

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import matplotlib.cm as cm
import numpy as np
import scipy.optimize as op

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

# note that you can do (x * ymix - x * ytrue)**2 or (ymix - ytrue)**2
# each has disadvantages.
# note magic number in the penalty for large variance
def badness_of_fit(lnpars, model, max_radius, log10_squared_deviation, radius_weighted):
    pars = np.exp(lnpars)
    x = np.arange(0.0005, max_radius, 0.001)
    residual = mixture_of_not_normals(x, pars) - hogg_model(x, model)
    if radius_weighted:
        numerator = np.sum(x * residual * residual)
        denominator = np.sum(x)
    else:
        numerator = np.sum(residual * residual)
        denominator = float(x.size)
    badness = numerator / denominator / 10.**log10_squared_deviation
    K = len(pars) / 2
    var = pars[K:]
    extrabadness = 0.0001 * np.sum(var) / max_radius**2
    return badness + extrabadness

def optimize_mixture(K, pars, model, max_radius, log10_squared_deviation, radius_weighted):
    newlnpars = op.fmin_bfgs(badness_of_fit, np.log(pars), args=(model, max_radius, log10_squared_deviation, radius_weighted))
    return (badness_of_fit(newlnpars, model, max_radius, log10_squared_deviation, radius_weighted), np.exp(newlnpars))

def plot_mixture(pars, prefix, model, max_radius, log10_squared_deviation, radius_weighted):
    x2 = np.arange(0.0005, np.sqrt(5. * max_radius), 0.001)**2 # note non-linear spacing
    y1 = hogg_model(x2, model)
    badness = badness_of_fit(np.log(pars), model, max_radius, log10_squared_deviation, radius_weighted)
    K = len(pars) / 2
    y2 = mixture_of_not_normals(x2, pars)
    plt.clf()
    plt.plot(x2, y1, 'k-')
    plt.plot(x2, y2, 'k-', lw=4, alpha=0.25)
    for k in range(K):
        plt.plot(x2, pars[k] * not_normal(x2, pars[k + K]), 'k-', alpha=0.5)
    plt.axvline(max_radius, color='k', alpha=0.25)
    badname = "badness"
    if radius_weighted: badname = "weighted " + badname
    plt.title(r"%s / $K=%d$ / max radius = $%.1f$ / %s = $%.3f\times 10^{%d}$"
              % (model, len(pars) / 2, max_radius, badname, badness * 100., log10_squared_deviation - 2.))
    plt.xlim(-0.1 * max_radius, 1.2 * max_radius)
    plt.ylim(-0.1 * 8.0, 1.1 * 8.0)
    plt.savefig(prefix + '.png')
    plt.loglog()
    plt.xlim(0.001, 5. * max_radius)
    plt.ylim(3.e-5, 1.5 * np.max(y1))
    plt.savefig(prefix + '_log.png')

def rearrange_pars(pars):
    K = len(pars) / 2
    indx = np.argsort(pars[K: K + K])
    amp = pars[indx]
    var = pars[K+indx]
    return np.append(amp, var)

# run this (possibly with adjustments to the magic numbers at top)
# to find different or better mixtures approximations
def main(input):
    model, radius_weighted = input
    max_radius = 8.0
    log10_squared_deviation = -4
    amp = np.array([1.0])
    var = np.array([1.0])
    pars = np.append(amp, var)
    (badness, pars) = optimize_mixture(1, pars, model, max_radius, log10_squared_deviation, radius_weighted)
    lastKbadness = badness
    bestbadness = badness
    for K in range(2, 20):
        prefix = '%s_K%02d_MR%02d' % (model, K, int(round(max_radius) + 0.01))
        if radius_weighted:
            prefix += "_w"
        print 'working on %s at K = %d (%s)' % (model, K, prefix)
        newvar = 2.0 * np.max(np.append(var, 1.0))
        newamp = 1.0 * newvar
        amp = np.append(newamp, amp)
        var = np.append(newvar, var)
        pars = np.append(amp, var)
        for i in range(2 * K):
            (badness, pars) = optimize_mixture(K, pars, model, max_radius, log10_squared_deviation, radius_weighted)
            if (badness < bestbadness) or (i == 0):
                print '%s %d %d improved' % (model, K, i)
                bestpars = pars
                bestbadness = badness
            else:
                print '%s %d %d not improved' % (model, K, i)
                var[0] = 2.0 * var[np.mod(i, K)]
                amp[0] = 0.5 * amp[np.mod(i, K)]
                pars = np.append(amp, var)
                if (bestbadness < 0.5 * lastKbadness) and (i > 4):
                    print '%s %d %d improved enough' % (model, K, i)
                    break
        lastKbadness = bestbadness
        pars = rearrange_pars(bestpars)
        amp = pars[0:K]
        var = pars[K:K+K]
        plot_mixture(pars, prefix, model, max_radius, log10_squared_deviation, radius_weighted)
        txtfile = open(prefix + '.txt', "w")
        txtfile.write("pars = %s" % repr(pars))
        txtfile.close
        if bestbadness < 1.0 and K > 15:
            break

if __name__ == '__main__':
    if True: # use multiprocessing
        from multiprocessing import Pool
        pmap = Pool(10).map
    else: # don't use multiprocessing
        pmap = map
    inputs = [('lup', True),
              ('exp', True),
              ('ser2', True),
              ('ser3', True),
              ('dev', True),
              ('ser5', True), ]
    pmap(main, inputs)
