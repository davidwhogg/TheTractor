import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from optimize_mixture_profiles import mixture_of_not_normals, hogg_model
    
def mixture_logprob(params, radii, target):
    residual = (mixture_of_not_normals(radii, params) - target) / 0.01
    # note radius (x) weighting, this makes the differential proportional to 2 * pi * r * dr
    chi_ish = np.sum(radii * residual**2)

    K = len(params)//2
    amps = params[:K]
    variances = params[K:]
    if np.any(variances < 0):
        return -np.inf
    amp_prior = -0.5 * np.sum(amps**2 / 10.**2)
    var_prior = -0.5 * np.sum(variances**2 / 10.**2)
    return -0.5 * chi_ish + amp_prior + var_prior

def opt_func(*args):
    return -mixture_logprob(*args)


def optimize_mixture(pars, radii, target):
    import scipy.optimize as opt
    # newpars = op.fmin_powell(opt_func, pars, args=(radii, target), maxfun=16384 * 2)
    # pars = 1. * newpars
    # newpars = op.fmin_bfgs(opt_func, pars, args=(radii, target), maxiter=128 * 2)
    # pars = 1. * newpars
    # newpars = op.fmin_cg(opt_func, pars, args=(radii, target), maxiter=128 * 2)
    # pars = 1. * newpars
    pars = opt.minimize(opt_func, pars, args=(radii, target), method='Powell').x
    pars = opt.minimize(opt_func, pars, args=(radii, target), method='BFGS').x
    pars = opt.minimize(opt_func, pars, args=(radii, target), method='CG').x
    pars = opt.minimize(opt_func, pars, args=(radii, target), method='Nelder-Mead').x
    return pars

# n: sersic index
# a: amps
# v: vars
def plot_one_row(n, a, v, xx):
    pars = np.append(a,v)
    mm = mixture_of_not_normals(xx, pars)
    K = len(a)
    for sp in [1,2,3]:
        plt.subplot(1,3,sp)
        if sp in [1,2]:
            plt.plot(xx, mm, '-', color='m', lw=5, label='Mixture');
            plt.plot(xx, hogg_model(xx, n), 'k-', lw=2, label='Sersic %.2f' % n)
        else:
            plt.plot(xx, xx*mm, '-', color='m', lw=5, label='Mixture');
            plt.plot(xx, xx*hogg_model(xx, n), 'k-', lw=2, label='Sersic %.2f' % n)
            
        ax = plt.axis()
        for ai,vi in zip(a, v):
            mi = mixture_of_not_normals(xx, np.array([ai,vi]))
            if sp == 3:
                mi *= xx
                
            if ai > 0:
                plt.plot(xx, mi, 'b-', alpha=0.5);
            else:
                plt.plot(xx, -mi, 'r-', alpha=0.5);
        if sp == 1:
            plt.yscale('log')
            plt.legend(loc='lower left')
        else:
            plt.legend(loc='upper right')
            #plt.axis(ax)
        plt.title('Sersic %.2f' % n)


def main():

    max_radius = 6.
    rstep = 0.001
    radii = np.arange(rstep/2., max_radius, rstep)

    dser = 0.01
    all_sersics = np.arange(0.6, 0.299, -dser)

    all_amps = []
    all_vars = []
    all_logprobs = []

    # (Re-)Initializations: iser -> (amps, vars)
    inits = dict([
        (0, ([8.5, 1.15], [0.83, 0.25])),
    # 0.5: one component only
        (np.argmin(np.abs(all_sersics - 0.5)),
         ([9.], [0.72])),
    # First one below 0.5: flip one amp negative.
        (np.argmin(np.abs(all_sersics - (0.5-dser))),
         ([9.24, -0.23], [0.71, 0.33])),
    # Switch to 3 components
    #(np.argmin(np.abs(all_sersics - 0.37)),
    #    ([33.85, -35.89, 10.25], [0.46, 0.37, 0.31])),
    #([15, -15, 10], [0.5, 0.35, 0.3])),
    #(np.argmin(np.abs(all_sersics - 0.39)),
    # ([31.13, -31.29, 8.43], [0.47, 0.38, 0.32])),
    #(np.argmin(np.abs(all_sersics - 0.40)),
    # ([31.34, -31.92, 8.92], [0.48, 0.39, 0.34])),
    (np.argmin(np.abs(all_sersics - 0.41)),
     ([25.57, -25.94, 8.78], [0.50, 0.40, 0.36])),
    # They cross over!
    #(np.argmin(np.abs(all_sersics - 0.42)),
    # ([24.05, -18.77, 3.19], [0.52, 0.43, 0.38])),
     ])

    for iser,sersic in enumerate(all_sersics):
        #for iser,sersic in enumerate(np.append(all_sersics, list(reversed(all_sersics[:-1])))):
        #all_ser2.append(sersic)
        target = hogg_model(radii, sersic)

        if iser in inits:
            a,v = inits[iser]
            initamps = np.array(a)
            initvars = np.array(v)
            print('Sersic index %.2f' % sersic, ': resetting initialization to',
                  initamps, initvars)

        print('Sersic %.2f' % sersic, 'initialized at amps', ', '.join(['%.2f'%a for a in initamps]), 'vars', ', '.join(['%.2f'%v for v in initvars]))
        pars = np.append(initamps, initvars)
        pars = optimize_mixture(pars, radii, target)
        fitamps = pars[:len(pars)//2].copy()
        fitvars = pars[len(pars)//2:].copy()
        all_amps.append(fitamps)
        all_vars.append(fitvars)
        all_logprobs.append(mixture_logprob(pars, radii, target))

        initamps = fitamps
        initvars = fitvars
        print('Sersic %.2f' % sersic, 'amps', ', '.join(['%.2f'%a for a in fitamps]), 'vars', ', '.join(['%.2f'%v for v in fitvars]))

    all_amps = np.array(all_amps)
    all_vars = np.array(all_vars)
    all_logprobs = np.array(all_logprobs)



    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    for i in range(3):
        J, = np.nonzero([len(a) > i for a in all_amps])
        plt.plot(all_sersics[J], np.array([all_amps[j][i] for j in J]), 'o-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture amplitudes')
    plt.axhline(0., color='k')
    plt.yscale('symlog')

    plt.subplot(1,2,2)
    for i in range(3):
        J, = np.nonzero([len(a) > i for a in all_vars])
        plt.plot(all_sersics[J], np.array([all_vars[j][i] for j in J]), 'o-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture variances');
    plt.suptitle('Gaussian Mixture approximations to Sersic profiles');
    plt.savefig('/tmp/mix.png')

    
    plt.figure(figsize=(16,4))

    xx = radii[radii <= 4]
    for i,(n,a,v,lnp) in enumerate(zip(all_sersics, all_amps, all_vars, all_logprobs)):
        plt.clf()
        plot_one_row(n, a, v, xx)
        #plt.suptitle('Logprob %.2f' % lnp)
        plt.savefig('/tmp/ser-%02i.png' % i)


if __name__ == '__main__':
    with np.errstate(all='ignore'):
        main()

    
