import numpy as np
from numba import jit
from bgunfolding.base import UnfoldingBase
from bgunfolding.ibu import IBU
from bgunfolding.metrics import chisq_sym, chisq_asym, emd, kl

class ImprIBU(UnfoldingBase):
    def __init__(self, 
                 n_mc, 
                 verbose = False):
        """
        Parameters
        ----------
        n_mc : int
            Number of MC iterations
            
        smoothing : str
            'polynomial' should be default
        smoothing_order : int
        smoothing_cut_overflow : boolean

        """
        
        super(UnfoldingBase, self).__init__()
        self.n_mc = n_mc
            
    def __repr__(object): 
        return 'improved ibu'
    
    def bayes(self, priorf, pec, cut_overflow = False):
        """
        Apply Bayes' Theorem to calculate inverse probabilites.
        
        Parameters
        ----------
        priorf : array-like
            Prior of f
        pec : ndarray
            Conditional probabilites P(E|C) (= response matrix)
        cut_overflow : bool

        Returns
        -------
        pce : ndarray
            inverse probabilites
            
        """
        
        if cut_overflow:
            pce = np.zeros((self.n_bins_true - 2, self.n_bins_est - 2))
            for i in range(self.n_bins_est - 2):
                pce[:,i] = pec[i] * priorf / sum(pec[i] * priorf)
                
            return pce

        else:
            pce = np.zeros((self.n_bins_true, self.n_bins_est))
            for i in range(self.n_bins_est):
                
                pce[:,i] = pec[i] * priorf / (sum(pec[i] * priorf) + self.b[i])

            return pce
    

    def apply_smearing(self, f_train):
        """
        Apply smearing to true density.
        
        Returns
        -------
        ni : ndarray
                
        """
        probs = self.A / self.A.sum(axis = 0)
        ni = np.zeros(self.A.shape)
        for i in range(self.n_bins_true):
            n_ = f_train[i]
            p = probs[:,i]
            ni[:,i] = np.random.multinomial(n_, p)
        
        self.ni = ni
        
        return ni
    
    def calc_alphas(self, ni, alphas_prior = 1):
        """
        Calculate alphas to use as parameter for Dirichlet distribution
        to model response matrix.
        """
        
        alphas = np.zeros(self.A.shape) + alphas_prior
        
        for i in range(self.n_bins_true):
            x = ni[:,i]
            alphas[:,i] += x
            
        return alphas
        
    def draw_response_and_efficiencies(self, alphas):
        """
        Draw columns of response matrix using Dirichlet distribution.
        
        Parameters
        ----------
        alphas : ndarray
        
        Returns
        -------
        response : ndarray
            response matrix
        """
        
        response = np.zeros(self.A.shape)
        eff = np.zeros(self.n_bins_true)
        
        for i in range(self.n_bins_true):
            # draw dirichlet for over and underflow as 
            # well to satisfy sum d = 1
            d = np.random.dirichlet(alphas[:,i])
            response[:,i] = d
            
            eff[i] = 1 - d[0] - d[-1]
            
        # efficiencies necessary?
        # p. 160 lista
        
        return response, eff
    
    def share_effects_to_causes(self, n, eff, inv_probs):
        """
        Observerations follow poisson distribution.
        Draw from prior conjugate (gamma).
        """
        # gamma distribution priors
        c = np.ones(self.n_bins_est)
        r = np.zeros(self.n_bins_est)

        c_fin = c + self.g # n = g - b
        
        # c_fin = c + self.g - self.b (if choose this: dont add bg in pce)
        # c_fin = c + self.g (add bg in denominator for pce/inv_probs)
        r_fin = r + 1
        
        evc = np.zeros(self.n_bins_true)
        for j in range(self.n_bins_est):
            # observation follow poisson distribution, draw from prior
            # conjugate (gamma)
            mu = np.random.gamma(shape = c_fin[j], scale = 1 / r_fin[j])
            
            # round
            m = np.round(mu) 
            if m < 1:
                m = 1

            # scaling factor
            scale = mu / m

            # draw from multinomial
            evcj = np.random.multinomial(m, inv_probs[:,j]) * scale

            # add results to expected causes
            evc += evcj  
            
        # take efficiency into account
        evc = evc / eff
        
        return evc
    
    def mc_unf(self, priorf, f_train, full_return = True):
        """
        Improved Iterative Bayesian Unfolding.
        """
        sx = np.zeros(self.n_bins_true)
        sxij = np.zeros((self.n_bins_true, self.n_bins_true))
        
        evcs = []
        responses = []
        n = self.g - self.b
        for k in range(self.n_mc):
            # apply smearing
            ni = self.apply_smearing(f_train)
            alphas = self.calc_alphas(ni)
            
            # draw response
            response, eff = self.draw_response_and_efficiencies(alphas)
            responses.append(response)
            
            # apply bayes theorem
            inv_probs = self.bayes(priorf, response)
            
            evc = self.share_effects_to_causes(n, eff, inv_probs)
            
            # evcs
            evcs.append(evc)
        
            sx += evc
            sxij += np.outer(evc, evc)

        # mean
        xm = sx / self.n_mc
        
        # cov, std, corrcoef
        covx = sxij / self.n_mc - np.outer(xm, xm)
        xs = np.sqrt(np.diag(covx))  
        rho = covx / np.outer(xs, xs)

        if full_return:
            return xm, xs, rho, covx, evcs, responses
        else:
            return xm, xs