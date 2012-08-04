# This file is part of The Tractor.

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
    rc('text', usetex=True)
import numpy as np
import pylab as plt
import pyfits
import cPickle as pickle

if __name__ == "__main__":
    kgimage = pyfits.open("kg/av_map_b12.fits")[0].data
    vmax = kgimage[np.where(np.isfinite(kgimage))].max()
    plt.gray()
    plt.clf()
    plt.imshow(kgimage, interpolation="nearest", vmin=0., vmax=vmax)
    plt.savefig("kg_b12.png")

    jdimage = pyfits.open("jd/brick15.medianebv.fits")[0].data
    plt.gray()
    plt.clf()
    plt.imshow(jdimage, interpolation="nearest", vmin=0., vmax=jdimage.max())
    plt.savefig("jd_b15.png")
