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
from optimize_mixture_profiles import badness_of_fit

def get_pars(model, K):
    MR = 8
    if model == "lux":
        MR = 4
    picklefn = "%s_K%02d_MR%02d.pickle" % (model, K, MR)
    picklefile = open(picklefn, "r")
    pars = pickle.load(picklefile)
    picklefile.close()
    return pars

def write_table_line(fd, strings):
    endofline = r"&"
    for i,string in enumerate(strings):
        if i + 1 == len(strings): endofline = r"\\"
        fd.write(r" %s %s" % (string, endofline) + "\n")
    return None

def write_table(model):
    tablefn = r"table_%s.tex" % model
    if model == "exp" or model == "lux":
        Ks = [4, 6, 8]
    else:
        Ks = [6, 8, 10]
    pars = [get_pars(model, K) for K in Ks]
    clist = "c|"
    for K in Ks: clist += "cc|"
    fd = open(tablefn, "w")
    fd.write(r"\begin{tabular}{%s}" % clist + "\n")
    fd.write(r"&" + "\n")
    fd.write(r"\multicolumn{%d}{|c|}{%s} \\" % (2 * len(Ks), model) + "\n")
    fd.write(r"\hline" + "\n")
    fd.write(r"$M^{\%s}=$ &" % model + "\n")
    write_table_line(fd, [r"\multicolumn{2}{|c|}{$%d$}" % K for K in Ks])
    fd.write(r"$m$ &" + "\n")
    write_table_line(fd, [r"$a^{\%s}_m$ & $\sqrt{v^{\%s}_m}$" % (model, model) for K in Ks])
    for k in range(np.max(Ks)):
        fd.write(r"$%d$ &" % (k + 1) + "\n")
        aks = []
        svks = []
        for i,K in enumerate(Ks):
            if k < K:
                aks += [r"%7.5f" % pars[i][k]]
                svks += [r"%7.5f" % np.sqrt(pars[i][K + k])]
            else:
                aks += [r"~"] # [r"\cdots"]
                svks += [r"~"] # [r"\cdots"]
        write_table_line(fd, [r"$%s$ & $%s$" % (a, sv) for a, sv in zip(aks, svks)])
    fd.write(r"\hline" + "\n")
    fd.write(r"$\sum_m a^{\%s}_m=$ &" % model + "\n")
    write_table_line(fd, [r"\multicolumn{2}{|c|}{$%6.3f$}" % q for q in [np.sum(pars[i][0:K]) for i, K in enumerate(Ks)]])
    fd.write(r"badness~$=$ &" + "\n")
    MR = 8.
    if model == "lux": MR = 4.
    badbases = []
    badcoeffs = []
    for i, K in enumerate(Ks):
        LSD = 0
        badness = 0.9
        while badness < 1.:
            LSD += np.floor(np.log10(badness)).astype(int)
            badness = badness_of_fit(np.log(pars[i]), model, MR, LSD)
        badbases += [LSD]
        badcoeffs += [badness]
    write_table_line(fd, [r"\multicolumn{2}{|c|}{$%4.2f\times 10^{%d}$}" % (c, b) for c, b in zip(badcoeffs, badbases)])
    fd.write(r"\end{tabular}" + "\n")
    fd.close()
    return None

if __name__ == "__main__":
    write_table("exp")
    write_table("ser2")
    write_table("ser3")
    write_table("dev")
    write_table("ser5")
    write_table("lux")
    write_table("luv")
