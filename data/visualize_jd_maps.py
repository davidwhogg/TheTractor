# This file is part of The Tractor.

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
    rc('text', usetex=True)
import pylab as plt
import pyfits

if __name__ == "__main__":
    jdimage = pyfits.open("jd/brick15.medianebv.fits")[0].data
    print jdimage.shape
    plt.gray()
    plt.clf()
    plt.imshow(jdimage)
    plt.savefig("jd_brick15.png")
