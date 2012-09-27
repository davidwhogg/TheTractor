'''
This file is part of the Tractor project.
Copyright 2011 David W. Hogg.
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
import os as os

models = ['exp', 'ser2', 'ser3', 'dev', 'ser5', 'lux', 'luv']
Ks = [4, 6, 8, 10]

def plot_mixtures_vs_n(K):
    return None

def plot_mixtures_vs_K(model):
    return None

def make_all_plots():
    for model in models:
        plot_mixtures_vs_K(model)
    for K in Ks:
        plot_mixtures_vs_n(K)
    return None

if __name__ == "__main__":
    make_all_plots()
