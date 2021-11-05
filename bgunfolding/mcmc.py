import numpy as np
import emcee

from joblib import Parallel, delayed
from multiprocessing import Pool

from bgunfolding.likelihood import llh_poisson, llh_tikhonov, log_prior
from bgunfolding.tikhonov_matrices import second_order_central
from bgunfolding.base import UnfoldingBase

from bgunfolding.utils import calc_glob_cc, MAD, calc_med_corr, calc_med_cov

def log_prob(f, g, b, A, acceptance, C, tau):
    lprior = log_prior(f)
    if not np.isfinite(lprior):
        return -np.inf
    
    lpoisson = llh_poisson(f, g, b, A, cut_overflow = False)
    ltikh = llh_tikhonov(f, C, tau, acceptance, cut_overflow = True)
    
    return lpoisson + ltikh + lprior


class MCMC(UnfoldingBase):
    
    def init_sampler(self, log_prob, nwalkers, ndim, C, tau, return_sampler = False):
        """
        Initialize Ensemble-Sampler Object.
        
        Parameters
        ----------
        log_prob : function
        nwalkers : int
        ndim : int
        tau : float
        """
        if self.is_fitted:
            ensemble_sampler = emcee.EnsembleSampler(nwalkers = nwalkers, 
                                                     ndim = ndim, 
                                                     log_prob_fn = log_prob,
                                                     args = [self.g, self.b, self.A, self.acceptance, C, tau]) # args regarding log_prob

            # store
            self.ensemble_sampler = ensemble_sampler
            self.log_prob = log_prob
            self.nwalkers = nwalkers
            self.ndim = ndim
            self.C = C
            
            if return_sampler:
                return ensemble_sampler
            
        else:
            print('Not yet fitted. Call \'fit\' method first.')


        
    def run_mcmc(self, p0, tau, nburnin, nmcmc, progress = False, return_chain = False):
        """
        Run MCMC.
        Chain will not be 
        
        Parameters
        ----------
        p0 : nd-array (ndim x nwalkers)
            Initial state for each walker
        nruns : int
            Number of runs to calculate mean autocorrelation time
        """
        # init sampler
        sampler = self.init_sampler(self.log_prob, self.nwalkers, self.ndim, self.C, tau, return_sampler = True)
        
        # burnin
        state = sampler.run_mcmc(p0, nburnin, progress = progress)
        sampler.reset()
        
        # ncmc
        sampler.run_mcmc(state, nmcmc, progress = progress)
        chain = sampler.get_chain(flat = True)
        
        # covariance, glob_cc
        cov = np.cov(chain.T)
        corr = np.corrcoef(chain.T)
        glob_cc = calc_glob_cc(cov)
        
        # robust cov, corrcoef
        med_cov = calc_med_cov(chain.T, n = self.n_bins_true)
        med_corr = calc_med_corr(chain.T, n = self.n_bins_true)
        med_glob_cc = calc_glob_cc(med_corr)
        
        # acor, acceptance
        acor = sampler.get_autocorr_time(tol = 0)
        acc_frac = sampler.acceptance_fraction
        
        # mean, std
        mean = np.mean(chain, axis = 0)
        std = np.std(chain, axis = 0)
        
        # percentiles
        quantiles = [16, 50, 84]
        percentiles = self._calc_percentiles(chain, self.ndim, quantiles = quantiles)
        
        # calculate p-value for this mcmc
        f_post = sampler.get_log_prob(flat = True)
        f_true_post = sampler.log_prob_fn(self.f)

        # calculate a p-value
        p_value = np.sum(f_true_post < f_post) / (nmcmc * self.nwalkers)
        
        d = {'percentiles': percentiles,
             'p_value': p_value,
             'cov': cov,
             'corr': corr,
             'glob_cc': glob_cc,
             'med_cov': med_cov,
             'med_corr': med_corr,
             'med_glob_cc': med_glob_cc,
             'tau': tau,
             'mean': mean,
             'std': std,
             'acor': acor,
             'acc_frac': acc_frac}
        
        if return_chain == True:
            return d, chain
        else:
            return d
   
    
    def init_parallelize_mcmc(self, ntau, p0s, taus, nburnin, nmcmc):
        """
        Setup parameters for parallel computation of MCMCs.
        """
        
        params = {'ntau': ntau,
                  'p0s': p0s,
                  'taus': taus,
                  'nburnin': nburnin,
                  'nmcmc': nmcmc}

        return params
        
    def parallelize_mcmc(self, n_jobs, params, progress = False, verbose = 11):
        """
        Run several MCMCs using parallel computation.
        """
        res = Parallel(n_jobs = n_jobs, backend = 'loky', verbose = verbose)(delayed(self.run_mcmc)(params['p0s'][i], params['taus'][i], params['nburnin'], params['nmcmc'], progress) for i in range(params['ntau']))
        
        return res
        
    def _calc_percentiles(self, chain, ndim, quantiles = [16, 50, 84]):
        """
        Calculate Percentiles from a flatten chain.
        """
        # take only ever acor[k]-th value
        percentiles = np.zeros((ndim, 3))
        for k in range(ndim):
            percentiles[k] = np.percentile(chain[:, k], quantiles)
    
        return percentiles
        
    def run_mcmc_convergence(self, p0, nburnin, nmcmc, check_interval, progress = False, threshold = 0.01, factor = 50):
        """
        Run MCMC with Convergence Check to estimate Autocorrelation for each chain/bin. 
        

        """
        # perform mcmc with convergence check
        acor = []
        acc_frac = []
        rel_err = []
        old_acor = np.inf
        
        # burnin
        state = self.ensemble_sampler.run_mcmc(p0, nburnin, progress = progress)
        self.ensemble_sampler.reset()
        
        for sample in self.ensemble_sampler.sample(state, iterations = nmcmc, progress = True):

            if self.ensemble_sampler.iteration % check_interval:
                continue

            # auto-correlation
            acor_check = self.ensemble_sampler.get_autocorr_time(tol = 0)
            acc_frac_check = self.ensemble_sampler.acceptance_fraction

            # append
            acor.append(acor_check)
            acc_frac.append(acc_frac_check)

            # check for convergence
            converged = np.all(acor_check * factor < self.ensemble_sampler.iteration)
            rel_err_check = np.abs(old_acor - acor_check) / acor_check
            rel_err.append(rel_err_check)
            
            converged &= np.all(rel_err_check < threshold)

            if converged:
                break

            # auto-correlation from last check
            old_acor = acor_check

        # convert to array
        acor = np.array(acor)
        acc_frac = np.array(acc_frac)

        # percentiles
        chain = self.ensemble_sampler.get_chain(flat = True)
        percentiles = self._calc_percentiles(chain, self.ndim, quantiles = [16, 50, 84])
        
        # mean, std
        mean = np.mean(chain, axis = 0)
        std = np.std(chain, axis = 0)
        
        # iteration
        iteration = self.ensemble_sampler.iteration
        print(f'chain converged (iteration : {iteration})')
        
        # store
        self.check_interval = check_interval
        
        d = {'acor': acor,
             'rel_err': rel_err,
             'iteration': iteration,
             'nmcmc': nmcmc,
             'check_interval': check_interval,
             'acc_frac': acc_frac,
             'mean': mean,
             'std': std,
             'percentiles': percentiles}
        
        return d