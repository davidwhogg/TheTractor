# This file is part of The Tractor.

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
    rc('text', usetex=True)
import pylab as plt
import pyfits
import cPickle as pickle

if __name__ == "__main__":
    fd = open("dstn/herschel-100-g100b.pickle")
    tractor = pickle.load(fd)
    fd.close()
    print tractor

if False:
    jdimage = pyfits.open("jd/brick15.medianebv.fits")[0].data
    print jdimage.shape
    print jdimage.min()
    print jdimage.max()
    plt.gray()
    plt.clf()
    plt.imshow(jdimage, interpolation="nearest", vmin=0., vmax=jdimage.max())
    plt.savefig("jd_brick15.png")
