import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from optimize_mixture_profiles import mixture_of_not_normals

def ser_model(radii, sersic):
    from optimize_mixture_profiles import hogg_model
    # Ramp down like the 'hogg_lux' model
    target = hogg_model(radii, sersic)
    inner = 3.
    outer = 4.
    target[radii > outer] *= 0.
    middle = (radii >= inner) * (radii <= outer)
    target[middle] *= (1. - ((radii[middle] - inner) / (outer - inner)) ** 2) ** 2
    return target

def mixture_loglikelihood(params, radii, target):
    residual = (mixture_of_not_normals(radii, params) - target) / 0.01
    # note radius (x) weighting, this makes the differential proportional to 2 * pi * r * dr
    chi_ish = np.sum(radii * residual**2)
    return -0.5 * chi_ish

def mixture_logprior(params):
    K = len(params)//2
    amps = params[:K]
    variances = params[K:]
    if np.any(variances < 0):
        return -np.inf
    amp_prior = -0.5 * np.sum(amps**2 / 10.**2)
    var_prior = -0.5 * np.sum(variances**2 / 10.**2)
    return amp_prior + var_prior

def mixture_logprob(params, radii, target):
    loglike = mixture_loglikelihood(params, radii, target)
    logprior = mixture_logprior(params)
    return loglike + logprior

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
    target = ser_model(xx, n)
    for sp in [1,2,3]:
        plt.subplot(1,3,sp)
        if sp in [1,2]:
            plt.plot(xx, mm, '-', color='m', lw=5, label='Mixture');
            plt.plot(xx, target, 'k-', lw=2, label='Sersic %.2f' % n)
        else:
            plt.plot(xx, xx*mm, '-', color='m', lw=5, label='Mixture');
            plt.plot(xx, xx*target, 'k-', lw=2, label='Sersic %.2f' % n)
            
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
            mx = max(max(mm), max(target))
            plt.ylim(mx*1e-6, mx*1.1)
            plt.legend(loc='lower left')
        else:
            plt.legend(loc='upper right')
            #plt.axis(ax)
        plt.title('Sersic %.2f' % n)

def fit_range(sersics, radii, init):
    a,v = init
    initamps = np.array(a)
    initvars = np.array(v)
    all_amps = []
    all_vars = []
    all_loglikes = []
    for sersic in sersics:
        target = ser_model(radii, sersic)
        #print('Sersic %.2f' % sersic, 'ini at',
        #      'amps', ', '.join(['%.4g'%a for a in initamps]),
        #      'vars', ', '.join(['%.4g'%v for v in initvars]))
        pars = np.append(initamps, initvars)
        pars = optimize_mixture(pars, radii, target)
        fitamps = pars[:len(pars)//2].copy()
        fitvars = pars[len(pars)//2:].copy()

        I = np.argsort(-fitamps)
        fitamps = fitamps[I]
        fitvars = fitvars[I]

        all_amps.append(fitamps)
        all_vars.append(fitvars)
        ll = mixture_loglikelihood(pars, radii, target)
        all_loglikes.append(ll)
        initamps = fitamps
        initvars = fitvars
        print('Sersic %.2f' % sersic, 'opt at',
              'amps', ', '.join(['%.4g'%a for a in fitamps]),
              'vars', ', '.join(['%.4g'%v for v in fitvars]))
        print('  log-likehood', ll)
    return all_amps, all_vars, all_loglikes

def main():

    max_radius = 4.
    rstep = 0.001
    radii = np.arange(rstep/2., max_radius, rstep)

    ### Need to fit parameters for both N-component mixtures at the
    ### transition indices.
    
    #dser = 0.01
    dser = 0.05
    #all_sersics = np.arange(0.6, 0.3-dser/2., -dser)
    #all_sersics = np.arange(1.0, 0.8-dser/2., -dser)
    #all_sersics = np.arange(0.85, 0.60-dser/2., -dser)
    #all_sersics = np.arange(0.75, 0.50-dser/2., -dser)
    #all_sersics = np.arange(0.4, 0.25-dser/2., -dser)
    # Approaching 0.25, |amps| become large -- exceeding 100
    
    all_sersics = []
    all_amps = []
    all_vars = []
    all_loglikes = []
    #all_logprobs = []

    dser = 0.05
    sersicsA = np.arange(1.0, 0.8-dser/2., -dser)
    init_1 = ([6.0, 4.34, 1.18, 0.223, 0.0308, 0.00235],
              [1.5, 0.461, 0.14, 0.0391, 0.00885, 0.0012])

    #dser = 0.025
    sersicsB = np.arange(0.8, 0.7-dser/2., -dser)
    #init_08 = ([5.729, 4.086, 0.8569, 0.1297, 0.01497],
    #           [1.28, 0.4863, 0.1643, 0.04959, 0.01203])
    init_08 = ([5.818, 4.1, 0.7976, 0.0947, 0.006378],
               [1.272, 0.4728, 0.1475, 0.03664, 0.005635])
    # At 0.65, seems to change mode -- amp goes negative, or pattern breaks

    #sersicsC = np.arange(0.7, 0.55-dser/2., -dser)
    sersicsC = np.arange(0.7, 0.6-dser/2., -dser)
    init_07 = ([6.09, 3.697, 0.4622, 0.02679],
               [1.108, 0.4562, 0.1305, 0.02194])
    #[5.845, 3.798, 0.5746, 0.0568],
    #[1.126, 0.484, 0.1605, 0.04168])
    # 0.55 -- reordered!

    sersicsC2 = np.arange(0.6, 0.55-dser/2., -dser)
    init_06 = ([6.09, 3.697, 0.4622],
               [1.108, 0.4562, 0.1305])
    
    dser = 0.01
    sersicsD = np.arange(0.55, 0.50+dser/2., -dser)
    #init_055 = ([6.522, 2.721, 0.1968],
    #            [0.873, 0.5162, 0.1796])
    #init_055 = ([7.747, 1.589, 0.04717],
    #            [0.8195, 0.4178, 0.08735])
    init_055 = ([7.747, 1.589],
                [0.8195, 0.4178])

    sersicsE = np.array([0.5])
    init_05 = ([9.065], [0.7213])

    dser = 0.01
    sersicsF = np.arange(0.5-dser, 0.45-dser/2., -dser)
    init_049 = ([9.24, -0.23], [0.71, 0.33])
    #init_049 = ([8.695, -0.4262, 0.007687], [0.7419, 0.4275, 0.08817])
    
    for sersics,ini,patch in [
        #(sersicsA, init_1,  False),
        #(sersicsB, init_08, False),
        # (sersicsC, init_07, False),
        (sersicsC2,init_06, False),
        (sersicsD, init_055, False),
        (sersicsE, init_05,  False),
        (sersicsF, init_049, True),
                        ]:
        amps,vars,loglikes = fit_range(sersics, radii, ini)

        # the one after 0.5: patch across 0.5
        if patch:
            amps_before = all_amps[-2]
            vars_before = all_vars[-2]

            amps_after = amps[0]
            vars_after = vars[0]

            amps_half = all_amps[-1]
            vars_half = all_vars[-1]
            assert(len(amps_before) == len(amps_after))

            print('vars before:', vars_before)
            print('vars after:', vars_after)
            print('vars half:', vars_half)
            
            patchvars = np.append([vars_half[0]], np.sqrt(vars_before[1:] * vars_after[1:]))
            print('Patch variances:', patchvars)

            patchamps = np.array([amps_half[0]] + [0.] * (len(amps_after)-1))
            print('Patch amps:', patchamps)
            
            #all_sersics.append(sersics[0])
            #all_amps.append(patchamps)
            #all_vars.append(patchvars)
            #all_loglikes.append(loglikes[0])

            # Replace 0.5 fit
            all_amps[-1] = patchamps
            all_vars[-1] = patchvars
            
        all_sersics.extend(sersics)
        all_amps.extend(amps)
        all_vars.extend(vars)
        all_loglikes.extend(loglikes)

        
    all_sersics = np.array(all_sersics)
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    #for i in range(3):
    for i in range(6):
        J, = np.nonzero([len(a) > i for a in all_amps])
        plt.plot(all_sersics[J], np.array([all_amps[j][i] for j in J]), 'o-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture amplitudes')
    plt.axhline(0., color='k')
    plt.yscale('symlog', linthreshy=1e-3)

    plt.subplot(1,2,2)
    #for i in range(3):
    for i in range(6):
        J, = np.nonzero([len(a) > i for a in all_vars])
        plt.plot(all_sersics[J], np.array([all_vars[j][i] for j in J]), 'o-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture variances');
    plt.yscale('log')
    plt.suptitle('Gaussian Mixture approximations to Sersic profiles');
    plt.savefig('/tmp/mix.png')
    
    return
    
    #all_sersics = np.arange(1.0, 0.25-dser/2, -dser)
    
    # (Re-)Initializations: iser -> (amps, vars)
    inits = dict([
    # 1.0 -- paper 1 values for lux, 6 components
    # (0, ([2.34853813e-03,   3.07995260e-02,   2.23364214e-01,
    #       1.17949102e+00,   4.33873750e+00,   5.99820770e+00],
    #      [1.20078965e-03,   8.84526493e-03,   3.91463084e-02,
    #       1.39976817e-01,   4.60962500e-01,   1.50159566e+00])),
    (np.argmin(np.abs(all_sersics - 1.0)),
     ([6.0, 4.34, 1.18, 0.223, 0.0308, 0.00235],
      [1.5, 0.461, 0.14, 0.0391, 0.00885, 0.0012])),
    # 0.85 -- one amplitude goes negative.  Smallest amplitude ~ 1e-3,
    # variances reasonably spaced
    # 5 components
    (np.argmin(np.abs(all_sersics - 0.85)),
     ([5.81, 4.2, 0.946, 0.148, 0.018], [1.35, 0.475, 0.152, 0.044, 0.0104])),
    # 0.6
    #(np.argmin(np.abs(all_sersics - 0.6)),
    # ([8.5, 1.15], [0.83, 0.25])),
    # Nothing wrong with the 5-component fit at 0.6; smallest component ~ 1e-3
    (np.argmin(np.abs(all_sersics - 0.6)),
     ([6.236, 3.169, 0.3216, 0.02589], [0.9646, 0.4898, 0.1714, 0.04615])),
    # Pattern changes at 0.55; all amps still positive, smallest ~ 1e-2.5
    (np.argmin(np.abs(all_sersics - 0.55)),
     ([6.522, 2.721, 0.1968], [0.873, 0.5162, 0.1796])),
     
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
    (np.argmin(np.abs(all_sersics - 0.40)),
     ([31.34, -31.92, 8.92], [0.48, 0.39, 0.34])),
    #(np.argmin(np.abs(all_sersics - 0.41)),
    # ([25.57, -25.94, 8.78], [0.50, 0.40, 0.36])),
    # They cross over!
    #(np.argmin(np.abs(all_sersics - 0.42)),
    # ([24.05, -18.77, 3.19], [0.52, 0.43, 0.38])),
    ])

    for iser,sersic in enumerate(all_sersics):
        #for iser,sersic in enumerate(np.append(all_sersics, list(reversed(all_sersics[:-1])))):
        #all_ser2.append(sersic)
        target = ser_model(radii, sersic)
        
        if iser in inits:
            a,v = inits[iser]
            initamps = np.array(a)
            initvars = np.array(v)
            print('Sersic index %.2f' % sersic, ': resetting initialization to',
                  initamps, initvars)

        print('Sersic %.2f' % sersic, 'initialized at amps', ', '.join(['%.4g'%a for a in initamps]), 'vars', ', '.join(['%.4g'%v for v in initvars]))
        pars = np.append(initamps, initvars)
        pars = optimize_mixture(pars, radii, target)
        fitamps = pars[:len(pars)//2].copy()
        fitvars = pars[len(pars)//2:].copy()
        all_amps.append(fitamps)
        all_vars.append(fitvars)
        #all_logprobs.append(mixture_logprob(pars, radii, target))
        all_loglikes.append(mixture_loglikelihood(pars, radii, target))

        initamps = fitamps
        initvars = fitvars
        print('Sersic %.2f' % sersic, 'amps', ', '.join(['%.4g'%a for a in fitamps]), 'vars', ', '.join(['%.4g'%v for v in fitvars]))
        #print('Sersic %.2f' % sersic, 'amps', ', '.join(['%g'%a for a in fitamps]), 'vars', ', '.join(['%g'%v for v in fitvars]))

    all_amps = np.array(all_amps)
    all_vars = np.array(all_vars)
    #all_logprobs = np.array(all_logprobs)
    all_loglikes = np.array(all_loglikes)


    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    #for i in range(3):
    for i in range(6):
        J, = np.nonzero([len(a) > i for a in all_amps])
        plt.plot(all_sersics[J], np.array([all_amps[j][i] for j in J]), 'o-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture amplitudes')
    plt.axhline(0., color='k')
    plt.yscale('symlog', linthreshy=1e-3)

    plt.subplot(1,2,2)
    #for i in range(3):
    for i in range(6):
        J, = np.nonzero([len(a) > i for a in all_vars])
        plt.plot(all_sersics[J], np.array([all_vars[j][i] for j in J]), 'o-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture variances');
    plt.yscale('log')
    plt.suptitle('Gaussian Mixture approximations to Sersic profiles');
    plt.savefig('/tmp/mix.png')

    plt.clf()
    plt.plot(all_sersics, all_loglikes, 'ko-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture log-likelihood');
    plt.suptitle('Gaussian Mixture approximations to Sersic profiles');
    mx = max(all_loglikes)
    plt.ylim(mx-200, mx)
    plt.savefig('/tmp/loglike.png')
    
    plt.figure(figsize=(16,4))

    xx = radii[radii <= 4]
    for i,(n,a,v) in enumerate(zip(all_sersics, all_amps, all_vars)):
        plt.clf()
        plot_one_row(n, a, v, xx)
        #plt.suptitle('Logprob %.2f' % lnp)
        plt.savefig('/tmp/ser-%02i.png' % i)


if __name__ == '__main__':
    with np.errstate(all='ignore'):
        main()

    
