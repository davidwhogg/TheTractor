"""
This file is part of the Tractor project.
Copyright 2011, 2012 David W. Hogg.

### bugs:
- duplicated code
- brittle code
"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import rc
rc("font",**{"family":"serif","size":12})
rc("text", usetex=True)
import pylab as plt
import numpy as np
import scipy.optimize as op
from multiprocessing import Pool
import cPickle as pickle
import os as os

models = ["exp", "ser2", "ser3", "dev", "ser5", "lux", "luv"]
Ks = [4, 6, 8, 10]

def get_pars(model, K):
    MR = 8
    if model == "lux":
        MR = 4
    picklefn = "%s_K%02d_MR%02d.pickle" % (model, K, MR)
    picklefile = open(picklefn, "r")
    pars = pickle.load(picklefile)
    picklefile.close()
    return pars

def plot_mixtures_vs_model(K):
    prefix = "mixtures_vs_model_K%02d" % K
    plt.figure(figsize=(5,5))
    plt.clf()
    for model in models[0:5]:
        pars = get_pars(model, K)
        amps = pars[0:K]
        sigmas = np.sqrt(pars[K:2 * K])
        plt.plot(sigmas, amps, "k-", alpha=0.5)
        plt.plot(sigmas, amps, "ko", ms=3.0)
        label = model
        tweak = 0.
        if model == "ser5": tweak = 5.
        if model == "dev": tweak = 1.
        plt.annotate(label, [sigmas[0], amps[0]], xytext=[-2,-10], textcoords="offset points", va="baseline", ha="center")
        plt.annotate(label, [sigmas[-1], amps[-1]], xytext=[4,-4+tweak], textcoords="offset points", va="baseline", ha="left")
    model = "ser"
    plt.xlabel(r"root-variance $\sqrt{v^{\mathrm{%s}}_m}$ (units of half-light radii)" % model)
    plt.ylabel(r"amplitudes $a^{\mathrm{%s}}_m$" % model)
    plt.title(r"approximations to ser profiles at $M^{\mathrm{ser}}=%d$" % K)
    plt.loglog()
    plt.xlim(0.5e-4, 2.e1)
    plt.ylim(0.5e-4, 2.e1)
    hogg_savefig(prefix + ".png")
    hogg_savefig(prefix + ".pdf")
    return None

def plot_mixtures_vs_K(model):
    prefix = "mixtures_vs_K_%s" % model
    plt.figure(figsize=(5,5))
    plt.clf()
    for K in Ks:
        pars = get_pars(model, K)
        amps = pars[0:K]
        sigmas = np.sqrt(pars[K:2 * K])
        plt.plot(sigmas, amps, "k-", alpha=0.5)
        plt.plot(sigmas, amps, "ko", ms=3.0)
        label = r"%d" % K
        tweak = 0.
        if model == "dev" or model == "luv":
            tweak = 4
            if K == 10:
                tweak = -2
        if model == "lux":
            tweak = 4 * (7 - K)
        plt.annotate(label, [sigmas[0], amps[0]], xytext=[-4,-4], textcoords="offset points", va="baseline", ha="right")
        plt.annotate(label, [sigmas[-1], amps[-1]], xytext=[4,-4+tweak], textcoords="offset points", va="baseline", ha="left")
    plt.xlabel(r"root-variance $\sqrt{v^{\mathrm{%s}}_m}$ (units of half-light radii)" % model)
    plt.ylabel(r"amplitudes $a^{\mathrm{%s}}_m$" % model)
    plt.title(r"approximations to %s" % model)
    plt.loglog()
    plt.xlim(0.5e-4, 2.e1)
    plt.ylim(0.5e-4, 2.e1)
    hogg_savefig(prefix + ".png")
    hogg_savefig(prefix + ".pdf")
    return None

def hogg_savefig(fn):
    print "writing %s" % fn
    return plt.savefig(fn)

def make_all_plots():
    for model in models:
        plot_mixtures_vs_K(model)
    for K in Ks:
        plot_mixtures_vs_model(K)
    return None

if __name__ == "__main__":
    make_all_plots()
