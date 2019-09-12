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

def mixture_logprior(params, extra=None):
    K = len(params)//2
    amps = params[:K]
    variances = params[K:]
    if np.any(variances < 0):
        return -np.inf

    # Standard deviation of the amplitude prior
    amp_std = 2.
    amp_prior = -0.5 * np.sum(amps**2 / amp_std**2)

    if extra:
        assert(len(extra) == len(amps))
        for a,sig in zip(amps, extra):
            if sig is None:
                continue
            # add extra amplitude priors!
            amp_prior += -0.5 * (a**2 / sig**2)

    return amp_prior

def mixture_logprob(*args):
    extra_priors = None
    if len(args) == 4:
        extra_priors = args[3]
    params, radii, target = args[:3]

    loglike  = mixture_loglikelihood(params, radii, target)
    logprior = mixture_logprior(params, extra=extra_priors)

    return loglike + logprior

def opt_func(*args):
    return -mixture_logprob(*args)

def optimize_mixture(pars, radii, target, priors=None):
    import scipy.optimize as opt
    # newpars = op.fmin_powell(opt_func, pars, args=(radii, target), maxfun=16384 * 2)
    # pars = 1. * newpars
    # newpars = op.fmin_bfgs(opt_func, pars, args=(radii, target), maxiter=128 * 2)
    # pars = 1. * newpars
    # newpars = op.fmin_cg(opt_func, pars, args=(radii, target), maxiter=128 * 2)
    # pars = 1. * newpars

    def callback(xk):
        print('  x', xk, '->', opt_func(xk, radii, target))

    #('BFGS', dict(eps=1e-6)),
    #('CG', {}),

    if priors is None:
        args = (radii, target)
    else:
        args = (radii, target, priors)
    
    for meth,kwargs in [('Powell', dict(ftol=1e-6, xtol=1e-6, maxfev=20000)),
                        ('Nelder-Mead', dict(maxfev=20000)),
                        ('Powell', dict(ftol=1e-6, xtol=1e-6, maxfev=20000)),
                        ('Powell', dict(ftol=1e-8, xtol=1e-8, maxfev=20000)),
        ]:
        R = opt.minimize(opt_func, pars, args=args, method=meth, options=kwargs)
        #, callback=callback)
        pars = R.x
        print('Method', meth, 'success', R.success, R.message, 'lnprob', -R.fun)
        #print('  Params', pars)
        #print('  Value', R.fun)
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

def fit_range(sersics, radii, init, priors=None):
    a,v = init
    initamps = np.array(a)
    initvars = np.array(v)
    all_amps = []
    all_vars = []
    all_loglikes = []
    all_logprobs = []

    if priors is not None:
        assert(len(priors) == len(sersics))
        for p in priors:
            assert(len(p) == len(a))

    for iser,sersic in enumerate(sersics):
        target = ser_model(radii, sersic)
        #print('Sersic %.2f' % sersic, 'ini at',
        #      'amps', ', '.join(['%.4g'%a for a in initamps]),
        #      'vars', ', '.join(['%.4g'%v for v in initvars]))
        pars = np.append(initamps, initvars)

        prior = None
        if priors is not None:
            prior = priors[iser]

        pars = optimize_mixture(pars, radii, target, priors=prior)
        #pars = sample_sersic(radii, target, pars)

        fitamps = pars[:len(pars)//2].copy()
        fitvars = pars[len(pars)//2:].copy()

        I = np.argsort(-fitamps)
        fitamps = fitamps[I]
        fitvars = fitvars[I]

        all_amps.append(fitamps)
        all_vars.append(fitvars)
        all_loglikes.append(mixture_loglikelihood(pars, radii, target))
        all_logprobs.append(mixture_logprob(pars, radii, target))
        initamps = fitamps
        initvars = fitvars
        print('Sersic %.2f' % sersic, 'opt at',
              'amps', ', '.join(['%.4g'%a for a in fitamps]),
              'vars', ', '.join(['%.4g'%v for v in fitvars]))
        print('  log-prob', all_logprobs[-1])
        print('(%.2f' % sersic, ', [', ''.join(['%g, '%a for a in fitamps]), '], [',
              ''.join(['%g, '%v for v in fitvars]), ']),')
    return all_amps, all_vars, all_loglikes

def sample_sersic(radii, target, pars):
    import emcee
    ndim, nwalkers = len(pars), 30
    scatter = 0.01 * np.ones(len(pars))
    params = emcee.utils.sample_ball(pars, scatter, size=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, mixture_logprob, args=[radii, target])
    nsteps = 5000
    sampler.run_mcmc(params, nsteps)

    # print('flatchain:', sampler.flatchain.shape)
    # for i in range(len(pars)):
    #     plt.clf()
    #     plt.plot(sampler.flatchain[:,i])
    #     plt.title('Parameter %i' % i)
    #     plt.savefig('chain-%i.png' % i)

    samples = sampler.flatchain[nsteps//2:,:]
    lnprobs = sampler.flatlnprobability[nsteps//2:]

    # import corner
    # fig = corner.corner(samples)
    # fig.savefig('corner.png')

    ibest = np.argmax(lnprobs)
    print('Largest log-prob:', lnprobs[ibest], 'at params', samples[ibest,:])
    print('Median params:', np.median(samples, axis=0))

    pbest = samples[ibest,:]
    p2 = optimize_mixture(pbest, radii, target)
    print('After opt:', mixture_logprob(p2, radii, target), 'at', p2)
    return p2

def main():

    max_radius = 4.
    rstep = 0.001
    radii = np.arange(rstep/2., max_radius, rstep)

    # emcee
    # init_04 = ([25.35, 8.269, -25.22], [0.5045, 0.3587, 0.4013])
    # sersic = 0.4
    # target = ser_model(radii, sersic)
    # initamps,initvars = init_04
    # pars = np.append(initamps, initvars)
    # sample_sersic(radii, target, pars)
    # return

    
    ### Need to fit parameters for both N-component mixtures at the
    ### transition indices (plus some overlap!)
    
    #dser = 0.01
    #dser = 0.05
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

    sersics7 = np.array([6.1, 6.2, 6.3])
    init_7 = ([ 9.78462, 5.92346, 3.08624, 1.37584,
                0.528399, 0.17315, 0.0471669, 0.0111288,  ],
                [ 1.52991, 0.132595, 0.0169103, 0.00245378,
                  0.000366884, 5.27404e-05, 6.76674e-06, 6.00242e-07, ])
    
    sersics6 = np.array([6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.7])
    init_6 = ([0.0113067,  0.04898785,  0.18195408,  0.55939775,
               1.46288372, 3.28556791,  6.27896305,  9.86946446],
               [6.07125356e-07,   7.02153046e-06,   5.60375312e-05,
                3.98494081e-04,   2.72853912e-03,   1.93601976e-02,
                1.58544866e-01,   1.95149972e+00])

    sersics3 = np.array([3.2, 3.1, 3., 2.5, 2., 1.5])
    init_3 = ([7.872, 5.073, 2.661, 1.112, 0.3659, 0.09262, 0.01655],
              [2.095, 0.3306, 0.06875, 0.01458, 0.002892, 0.0004967, 6.458e-05])

    sersics15 = np.array([1.55, 1.51, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])
    init_15 = ([6.653, 4.537, 1.776, 0.5341, 0.121, 0.01932],
               [1.83, 0.4273, 0.1187, 0.03255, 0.007875, 0.001491])
    
    sersicsA = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.76, 0.75])
    init_1 = ([6.0, 4.34, 1.18, 0.223, 0.0308, 0.00235],
              [1.5, 0.461, 0.14, 0.0391, 0.00885, 0.0012])

    sersicsB = np.array([0.81, 0.8, 0.75, 0.7])
    init_08 = ([5.818, 4.1, 0.7976, 0.0947, 0.006378],
               [1.272, 0.4728, 0.1475, 0.03664, 0.005635])
    # At 0.65, seems to change mode -- amp goes negative, or pattern breaks

    # From 0.6 down to 0.55, the two largest-variance components merge
    # (sharp change at about 0.56)
    sersicsC = np.array([0.76, 0.75, 0.71, 0.7, 0.65, 0.6, 0.58, 0.57])
    init_07 = ([ 6.06735, 3.75046, 0.485251, 0.0287147,  ],
               [ 1.12455, 0.455509, 0.12943, 0.0216252,  ])

    dser = 0.01
    sersicsD = np.arange(0.61, 0.50+dser/2., -dser)
    init_055 = ([7.747, 1.589],
                [0.8195, 0.4178])

    sersicsE = np.array([0.5])
    init_05 = ([9.065], [0.7213])

    dser = 0.01
    sersicsF = np.arange(0.5-dser, 0.4-dser/2., -dser)
    init_049 = ([9.24, -0.23], [0.71, 0.33])

    sersicsG = np.arange(0.42, 0.29-dser/2., -dser)
    init_04 = ([ 13.5581, 0.433216, -5.44774,  ], [ 0.575113, 0.304652, 0.385431,  ])

    # sersicsFG = np.array([0.40, 0.41, 0.42, 0.43])
    # initFG = ([ 15.4473, 1.43828, -8.47514,  ], [ 0.540383, 0.28262, 0.365116,  ])
    # priorsFG = [ [None, 1., None], [None, 0.5, None], [None, 0.25, None], [None, 0.125, None] ]
    sersicsFG = np.array([0.39, 0.40, 0.41, 0.42])
    initFG = ([ 15.4473, 1.43828, -8.47514,  ], [ 0.540383, 0.28262, 0.365116,  ])
    priorsFG = [ [None, 1., None], [None, 0.5, None], [None, 0.25, None], [None, 0.125, None] ]
    
    #sersicsG = np.array([0.42, 0.43])
    #init_04 = ([ 13.5581, 0.433216, -5.44774,  ], [ 0.575113, 0.304652, 0.385431,  ])
    
    for sersics,ini,priors,patch in [
        # (sersics7, init_7,   None, False),
        # (sersics6, init_6,   None, False),
        # (sersics3, init_3,   None, False),
        # (sersics15, init_15, None, False),
        # (sersicsA, init_1,   None, False),
        # (sersicsB, init_08,  None, False),
        # (sersicsC, init_07,  None, False),
        # (sersicsD, init_055, None, False),
        # (sersicsE, init_05,  None, False),
        # (sersicsF, init_049, None, True),
        # (sersicsF, init_049, None, False),
        # (sersicsG, init_04, None, False),
        (sersicsFG, initFG, priorsFG, False),
        ]:
        amps,vars,loglikes = fit_range(sersics, radii, ini, priors=priors)

        # amps2,vars2,loglikes2 = fit_range(reversed(sersics), radii,
        #                                   (amps[-1], vars[-1]))
        # 
        # amps3,vars3,loglikes3 = fit_range(sersics, radii,
        #                                   (amps2[-1], vars2[-1]))
        # 
        # amps4,vars4,loglikes4 = fit_range(reversed(sersics), radii,
        #                                   (amps3[-1], vars3[-1]))
        # amps,vars = amps4,vars4
        
        print('After fitting range forward:', sersics)
        #print('amps:', amps)
        #print('vars:', vars)
        print('log-likes:', loglikes)

        # print('After fitting range backwards:', sersics)
        # #print('amps:', amps2)
        # #print('vars:', vars2)
        # print('log-likes:', loglikes2)
        # 
        # print('After fitting range forwards again:', sersics)
        # print('log-likes:', loglikes3)
        # 
        # print('After fitting range backwards again:', sersics)
        # print('log-likes:', loglikes4)

        for sersic,fitamps,fitvars in reversed(list(zip(sersics,amps,vars))):
            print('(%.2f' % sersic, ', [', ''.join(['%g, '%a for a in fitamps]), '], [',
                  ''.join(['%g, '%v for v in fitvars]), ']),')

        
        # the one after 0.5: patch across 0.5
        if patch:
            amps_before = all_amps[-2]
            vars_before = all_vars[-2]

            amps_after = amps[0]
            vars_after = vars[0]

            amps_half = all_amps[-1]
            vars_half = all_vars[-1]
            assert(len(amps_before) == len(amps_after))

            #print('vars before:', vars_before)
            #print('vars after:', vars_after)
            #print('vars half:', vars_half)
            
            patchvars = np.append([vars_half[0]], np.sqrt(vars_before[1:] * vars_after[1:]))
            #print('Patch variances:', patchvars)

            patchamps = np.array([amps_half[0]] + [0.] * (len(amps_after)-1))
            #print('Patch amps:', patchamps)
            
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
    for i in range(8):
        J, = np.nonzero([len(a) > i for a in all_amps])
        plt.plot(all_sersics[J], np.array([all_amps[j][i] for j in J]), 'o-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture amplitudes')
    plt.axhline(0., color='k')
    plt.yscale('symlog', linthreshy=1e-3)

    plt.subplot(1,2,2)
    #for i in range(3):
    for i in range(8):
        J, = np.nonzero([len(a) > i for a in all_vars])
        plt.plot(all_sersics[J], np.array([all_vars[j][i] for j in J]), 'o-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture variances');
    plt.yscale('log')
    plt.suptitle('Gaussian Mixture approximations to Sersic profiles');
    plt.savefig('/tmp/mix.png')

    plt.xscale('log')
    xt = [6, 5, 4, 3, 2, 1, 0.5, 0.4, 0.3]
    plt.xticks(xt, ['%g'%x for x in xt])
    plt.subplot(1,2,1)
    plt.xscale('log')
    plt.xticks(xt, ['%g'%x for x in xt])
    plt.savefig('/tmp/mix2.png')

    plt.clf()
    plt.subplot(1,2,1)
    for s,a,v in zip(all_sersics, all_amps, all_vars):
        plt.plot(s + np.zeros_like(a), a, '-')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture amplitudes')
    plt.axhline(0., color='k')
    plt.yscale('symlog', linthreshy=1e-3)
    plt.xscale('log')
    xt = [6, 5, 4, 3, 2, 1, 0.5, 0.4, 0.3]
    plt.xticks(xt, ['%g'%x for x in xt])
    plt.subplot(1,2,2)
    for s,a,v in zip(all_sersics, all_amps, all_vars):
        plt.plot(s + np.zeros_like(v), v, '.')
    plt.xlabel('Sersic index')
    plt.ylabel('Mixture variances');
    plt.yscale('log')
    plt.xscale('log')
    xt = [6, 5, 4, 3, 2, 1, 0.5, 0.4, 0.3]
    plt.xticks(xt, ['%g'%x for x in xt])
    plt.savefig('/tmp/mix3.png')
    
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

    
