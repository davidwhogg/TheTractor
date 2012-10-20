"""
This file is part of the Tractor project.
Copyright 2012 David W. Hogg.

### bugs:
- brittle code
- should import some "python latex table-writing" module
"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import rc
rc("font",**{"family":"serif","size":12})
rc("text", usetex=True)
import numpy as np
import cPickle as pickle

def get_pars(model, K):
    MR = 8
    if model == "lux":
        MR = 4
    picklefn = "%s_K%02d_MR%02d.pickle" % (model, K, MR)
    picklefile = open(picklefn, "r")
    pars = pickle.load(picklefile)
    picklefile.close()
    return pars

def write_table_line(strings):
    endofline = r"&"
    for i,string in enumerate(strings):
        if i + 1 == len(strings): endofline = r"\\"
        print r" %s %s" % (string, endofline)
    return None

def write_table(model):
    if model == "exp" or model == "lux":
        Ks = [4, 6, 8]
    pars = [get_pars(model, K) for K in Ks]
    clist = "c|"
    for K in Ks: clist += "cc|"
    print r"\begin{tabular}{%s}" % clist
    print r"&"
    print r"\multicolumn{%d}{|c|}{%s} \\" % (2 * len(Ks), model)
    print r"\hline"
    print r"$M^{\%s}=$ &" % model
    write_table_line([r"\multicolumn{2}{|c|}{$%d$}" % K for K in Ks])
    print r"$m$ &"
    write_table_line([r"$a^{\%s}_m$ & $\sqrt{v^{\%s}_m}$" % (model, model) for K in Ks])
    for k in range(np.max(Ks)):
        print r"$%d$" % (k + 1)
        aks = []
        svks = []
        for i,K in enumerate(Ks):
            if k < K:
                aks += [r"%7.5f" % pars[i][k]]
                svks += [r"%7.5f" % np.sqrt(pars[i][2 * k])]
            else:
                aks += [r"\cdots"]
                svks += [r"\cdots"]
        write_table_line([r"$%s$ & $%s$" % (a, sv) for a, sv in zip(aks, svks)])
    print r"\hline"
    print r"$\sum_m a^{\%s}_m=$ &" % model
    write_table_line([r"\multicolumn{2}{|c|}{$%7.5f$}" % q for q in [np.sum(pars[i][0:K]) for i,K in enumerate(Ks)]])
    print r"badness\,$=$ &"
    write_table_line([r"\multicolumn{2}{|c|}{$%5.3f\times 10^{%d}$}" % (0, 0) for K in Ks])
    print r"\end{tabular}{%s}"
    return None

if __name__ == "__main__":
    write_table("exp")
