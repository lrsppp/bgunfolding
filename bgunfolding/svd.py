import numpy as np
import matplotlib.pyplot as plt
from bgunfolding.base import UnfoldingBase
from bgunfolding.tikhonov_matrices import *
from scipy.optimize import minimize
import numdifftools as nd
from joblib import Parallel, delayed

class SVD(UnfoldingBase):
    def __init__(self, C = 'second_order', weighted = False):
        '''
        C: str
            Tikhonov matrix. Choose between: `second_order`, `identity`, `selective_identity`
            second_order is default. Forces flat spectrum.
        '''
        super(UnfoldingBase, self).__init__()
        self.C = C
        self.weighted = weighted
    
    def __repr__(self):
        return 'svd'
    
    def _parallel_estimate_tau(self, i, tau):
        f_est, cov, lx, ly = self.predict(tau)
        glob_cc = self.calc_global_cc(cov)
        
        return f_est, cov, lx, ly, glob_cc
        
    def estimate_tau(self, tau_min, tau_max, n_tau, n_jobs = 2):
        """
        Parameters
        ----------
        tau_min : float
            10**tau_min
        tau_max : float
            10**tau_max
        n_tau : int
        
        Returns
        -------
        tau_est : float
        L1 : array-like
        L2 : array-like
        
        """
        tau_space = np.logspace(tau_min, tau_max, n_tau)
        f_est_array = np.zeros((n_tau, self.n_bins_true))
        cov_array = np.zeros((n_tau, self.n_bins_true, self.n_bins_true))
        
        glob_cc = np.zeros(n_tau)
        LX = np.zeros(n_tau)
        LY = np.zeros(n_tau)
        
        # parallelize
        r = Parallel(n_jobs = n_jobs, backend = 'loky', verbose = 0)(delayed(self._parallel_estimate_tau)(i, tau) for i, tau in enumerate(tau_space))
        
        # store
        f_est_array = np.array([r_[0] for r_ in r])
        cov_array = np.array([r_[1] for r_ in r])
        LX = np.array([r_[2] for r_ in r])
        LY = np.array([r_[3] for r_ in r])
        glob_cc = np.array([r_[4] for r_ in r])
        
        nan_idx = np.isnan(glob_cc)
        glob_cc_ = glob_cc[nan_idx == False]
        tau_est = tau_space[np.where(glob_cc_ == np.min(glob_cc_))[0][0]]
        self.tau_est = tau_est
        
        d = {'tau_est': tau_est,
             'cov_array': cov_array,
             'tau_space': tau_space,
             'glob_cc': glob_cc,
             'nan_idx': nan_idx,
             'f_est_array': f_est_array}
        
        # l-curve criterion
        d_curv = self.curvature_criterion(tau_space, LX, LY)
        d.update(d_curv)
    
        return d
    
    def curvature_criterion(self, tau_space, lx, ly):
        """
        Find maximum of L-Curve curvature.
        """
        rho = np.log(lx)
        xi = np.log(ly)
        
        drho = np.gradient(rho)
        ddrho = np.gradient(drho)

        dxi = np.gradient(xi)
        ddxi = np.gradient(dxi)

        curv = 2 * (drho * ddxi - ddrho * dxi) / (drho**2 + dxi**2)**(3/2)
        max_idx = np.where(curv == np.max(curv))[0][0]
        
        tau_est_curv = tau_space[max_idx]
        curv_max = curv[max_idx]

        # l curve max curvature
        rho_max = rho[max_idx]
        xi_max = xi[max_idx]
        
        d = {'rho': rho,
             'xi': xi,
             'lx': lx,
             'ly': ly,
             'drho': drho,
             'ddrho': ddrho,
             'dxi': dxi,
             'ddxi': ddxi,
             'curv': curv,
             'max_idx': max_idx, 
             'tau_est_curv': tau_est_curv,
             'curv_max': curv_max,
             'rho_max': rho_max,
             'xi_max': xi_max}
        
        return d
    
    
    def calc_global_cc(self, cov):
        return np.mean(np.sqrt(1 - 1 / ( np.diag(cov) * np.diag(np.linalg.inv(cov)))))
    
    def svd(self):
        """
        
        """
        
        if self.is_fitted == True:
            self.u, self.s, self.vh = np.linalg.svd(self.A)
            
            return self.u, self.s, self.vh
        else:
            print('Not fitted yet.')
            
    def calc_filter_factors(self, tau, s):
        filter_factors = s**2 / (s**2 + tau**2)

        return filter_factors
        
    def predict(self, tau):
        """
        Parameters
        ----------
        tau : float
            Regularization Parameter
            
        Returns
        -------
        f_est : array-like
        cov : array-like
        resid : float
            Represents Lx in an L-Curve plot.
        regul : float
            Represents Ly in an L-Curve plot.
        """
        
        if tau == None:
            raise Exception('Regularization Parameter not defined (None-Type).')
        
        # Weighted and regularized least squares fit
        cov_n = self.cov_n
        cov_n_inv = np.linalg.pinv(cov_n)
        # https://stats.stackexchange.com/questions/52704/covariance-of-linear-regression-coefficients-in-weighted-least-squares-method
        # Tikhonov Matrices
        if self.C == 'second_order':
            # second order derivative linear operator
            self.rmatrix= second_order_central(len(self.f) - 2) # cut off over- and underflow

        elif self.C == 'identity':
            # identity matrix
            self.rmatrix = np.eye(len(self.f) - 2)
            
        elif self.C == 'selective_identity':
            # svd
            u, s, vt = self.svd()
            x = tau**2 - s**2

            # no damping of solution components with small index
            self.rmatrix = np.diag(np.max(np.diag(x[1:-1]), 0))
            
        # minimize
        x0 = np.ones(self.n_bins_true)
        bounds = [[1e-4, None] for i in range(self.n_bins_true)]

        res = minimize(self.residuals, x0, bounds = bounds, args = (tau, self.weighted))
        
        # estimated density
        f_est = res.x
        
        hess = nd.Hessian(self.residuals, method = 'complex')
        cov = np.linalg.inv(hess(f_est, tau, self.weighted))
        
         # l curve (unweighted )
        lx = np.linalg.norm(self.A @ f_est + self.b - self.g)
        ly = np.linalg.norm(self.rmatrix @ np.log10(f_est / self.acceptance)[1:-1])
    
        self.f_est = f_est
        self.cov = cov

        return f_est, cov, lx, ly
    
    
    def residuals(self, f, tau, weighted = False):
        
        self.W = np.linalg.inv(np.diag(self.g + self.b))
        
        if f.any() <= 0:
            return np.inf
        else:
            f_eff = np.log10(f[1:-1] / self.acceptance[1:-1])

            # weight residuals with errors
            if weighted:
                S = (self.A @ f + self.b - self.g).T @ self.W @ (self.A @ f + self.b - self.g) + tau**2 * (self.rmatrix @ f_eff).T @ (self.rmatrix @ f_eff)
            else:
                S = (self.A @ f + self.b - self.g).T @ (self.A @ f + self.b - self.g) + tau**2 * (self.rmatrix @ f_eff).T @ (self.rmatrix @ f_eff)

            return S
    
    def estimate_minimum(self, tau_space, glob_cc):
        try:
            tau_est = tau_space[np.where(glob_cc == np.min(glob_cc))[0][0]]
            return tau_est
        except:
            print('Could not estimate regularization parameter.')
            return None
    
    def plot_glob_cc(self):
        
        plt.plot(self.tau_space, self.glob_cc, label = 'Mean of Global Correlation Coefficients')
        plt.xscale('log')
        plt.xlabel(r'$\mathrm{Regularization\,Parameter}\,\tau$')
        plt.ylabel(r'$\hat{\rho}_{\tau}$')
        plt.legend(loc = 'best')
        plt.tight_layout()