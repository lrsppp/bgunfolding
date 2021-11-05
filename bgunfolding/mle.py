import numpy as np
from bgunfolding.base import UnfoldingBase
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from bgunfolding.likelihood import llh_poisson, llh_tikhonov, hess_poisson, hess_tikhonov


class MLE(UnfoldingBase):
    
    def __init__(self, C, x0, bounds = None):
        """
        C : array-like
            Tikhonov Matrix used for regularization. Default should be discrete second ordel
            central derivative.
            
        x0 : array-like
            Prior for minimzation
            
        bounds : list
            Bounds for each entry of estimated density f.
        """
        super(UnfoldingBase, self).__init__()
        self.C = C
        self.x0 = x0
        self.bounds = bounds
        self.cut_overflow_poisson = False
        self.cut_overflow_tikhonov = True
    
    def __repr__(self):
        return 'plu'
    
    def predict(self, tau):
        """
        Parameters
        ----------
        tau : float
            Regularization parameter
            
        Returns
        -------
        f_est : array-like
            Estimated density.
        """
        
        if tau is None:
            print(type(tau))
            print(f'No valid regularization parameter.')
        
        else:
        
            function = lambda f_est, tau: - llh_poisson(f_est, self.g, self.b, self.A, cut_overflow = self.cut_overflow_poisson)\
                                        - llh_tikhonov(f_est, self.C, tau, self.acceptance, cut_overflow = self.cut_overflow_tikhonov)

            if self.is_fitted == True:
                res = minimize(function, 
                               x0 = self.x0,
                               bounds = self.bounds,
                               args = (tau))
                return res.x

            else:
                print('Not fitted yet.')
    
    def predict_hess(self, f_est, tau):
        """
        Parameters
        ----------
        f_est : array-like
            Estimated density f_est
        tau : float
            Regularization parameter
        
        """

        res = hess_poisson(f_est, self.g, self.b, self.A) +\
              hess_tikhonov(f_est, self.C, tau, self.acceptance)

        return res

        
    def estimate_tau(self, tau_min, tau_max, n_tau = 250, log = True):
        """
        Does a scan of tau parameters within a logspace starting from 10^tau_min to 10^tau_max.

        For each tau the predict method is called and the corresponding hessian matrix
        is calculated. These are used to calculate the global correlation coefficients (glob_cc).
        
        Parameters
        ----------
        tau_min : int
            Minimum exponent that defines the logspace (10^tau_min).
            
        tau_max: int
            Maximum exponent that defines the logspace (10^tau_max).
            
        n_tau : int
            Number of evenly spaced floats within the logspace.
        """
        
        if log == True:
            tau_space = np.logspace(tau_min, tau_max, n_tau)
        elif log == False:
            tau_space = np.linspace(tau_min, tau_max, n_tau)
        
        glob_cc = np.zeros(n_tau)
        hess = np.zeros((n_tau, self.n_bins_true - 2, self.n_bins_true - 2))
        
        for i, tau in enumerate(tau_space):
            res = self.predict(tau)
            hess[i] = self.predict_hess(res, tau)
            glob_cc[i] = self.calc_glob_cc(hess[i])
            
            self.x0 = res
        
        self.tau_est = self.estimate_minimum(tau_space, glob_cc)
        self.hess = hess
        self.glob_cc = glob_cc
        self.tau_space = tau_space
        
        return self.tau_est
        
    def calc_glob_cc(self, cov):
        """
        Calculate mean of global ccorrelation coefficients
        
        \rho_j = \sqrt{1 - [(V_x)_{jj} \cdot (V_x)_{jj}^{-1}]^{-1}}        
        
        Parameters
        ----------
        cov : ndarray
            Covariance Matrix
            
        Returns : float
            Global Mean Correlation Coefficients
        
        """
        
        glob_cc = np.mean(np.sqrt(1 - 1 / ( np.diag(cov) * np.diag(np.linalg.inv(cov)))))
        
        return glob_cc

    def estimate_minimum(self, tau_space, glob_cc):
        """
        Estimate regularization parameter tau which correpsonds to minimum of
        global correlation coefficients.
        
        Parameters
        ----------
        tau_space : array of length N
        
        glob_cc : array of length N
        """
        
        try:
            tau_est = tau_space[np.where(glob_cc == np.min(glob_cc))[0][0]]
            return tau_est
        except:
            print('Could not estimate regularization parameter.')
            
    def plot_glob_cc(self):
        """
        Helper function to quickly plot mean of global correlation coefficients versus regularization
        parameter tau.
        """
        
        plt.plot(self.tau_space, self.glob_cc, label = 'Mean of Global Correlation Coefficients')
        plt.xscale('log')
        plt.xlabel(r'$\mathrm{Regularization\,Parameter}\,\tau$')
        plt.ylabel(r'$\hat{\rho}_{\tau}$')
        plt.legend(loc = 'best')
        plt.tight_layout()
