import numpy as np
from numba import jit
from bgunfolding.base import UnfoldingBase
from bgunfolding.metrics import chisq_sym, chisq_asym, emd, kl

def smooth_polynomial(arr, acceptance, order, cut_overflow = False, return_params = False):
    """
    Fit a polynomial of order order to transformed estimated density f_est.
    A first-order polynomial fit corresponds to a powerlaw fit in a log-plot.
    A second-order (parabola) polynomial fit corresponds to a logparabola fit in log-plot
    """
    
    # https://arxiv.org/pdf/1806.03350.pdf
    # Transform
    
    x = np.arange(len(arr))
    y = np.log10(arr / acceptance)
    
    if cut_overflow:
        params = np.polyfit(x[1:-1], y[1:-1], deg = order)
        
    else:
        params = np.polyfit(x, y, deg = order)
        
    y_smoothed = np.polyval(params, x)
    arr_smoothed = 10**y_smoothed * acceptance
    
    if return_params:
        return arr_smoothed, params
    else:
        return arr_smoothed

class IBU(UnfoldingBase):
    def __init__(self, 
                 n_iterations, 
                 x0, 
                 epsilon = 1e-6, 
                 metric = 'chisq_sym',
                 smoothing = None, 
                 smoothing_order = None,
                 smoothing_cut_overflow = False,
                 convergence_break = True,
                 verbose = False):
        """
        Parameters
        ----------
        n_iterations : int
            Number of iterations
            
        epsilon : float
            minimum metric distance between iterations.
            
        x0 : array-like
            Prior
            
        smoothing : function
            A function (f) -> (f_smooth) which smoothes each estimate f_est.
            
        metric : function
            A function (f_est, f_true) -> () which calculates distance between 
            estimated density and previous estimate. Default is symmetric Chi Square
            Distance.
            
        convergence_break : boolean
            Iterative process will be interrupted if metric distance is smaller than
            epsilon. 
        """
        
        super(UnfoldingBase, self).__init__()
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.x0 = x0
        self.metric = metric
        self.convergence_break = convergence_break
        self.smoothing = smoothing
        self.smoothing_order = smoothing_order
        self.smoothing_cut_overflow = smoothing_cut_overflow
        
        self.verbose = verbose
        if self.smoothing != None:
            self.is_smoothed = True
        else:
            self.is_smoothed = False
    
    def __repr__(object): 
        return 'ibu'
    
    def predict(self):
        """
        See: Lista L., Statistical Methods For Data Analysis (2017), p. 170
        """
        if self.is_fitted == True:
            # define prior
            f_est = self.x0
            cov = np.zeros((self.n_bins_true, self.n_bins_true))
            
            # error (default: chi square distance between estimated and true density)
            self.error = np.inf
            
            # covariance matrix of g - b
            self.n = self.g - self.b
            self.cov_n = np.diag(self.g + self.b) # skellam distribution (difference of two poisson)
            
            for iteration in range(1, self.n_iterations+1):
                
                # smoothing
                if self.smoothing is not None and iteration > 1:
                    if self.smoothing == 'polynomial':
                        f_smooth = smooth_polynomial(f_est, 
                                                     acceptance = self.acceptance, 
                                                     order = self.smoothing_order,
                                                     cut_overflow = self.smoothing_cut_overflow)
                        
                    else:
                        print(f'No smoothing function has been found under the name {self.smoothing}.')
                else:
                    f_smooth = f_est
                    
                # unsmoothed estimate from previous iteration
                f_est_prev = f_est

                # calculate unfolding matrix
                M = (self.A * f_smooth).T / (np.sum(self.A * f_smooth, axis = 1) + self.b) 
                
                # bayes unfolding
                f_est = np.sum(M * self.g, axis = 1)[:]
                
                # error propagation
                if iteration == 1:
                    dfdn = M
                
                else:
                    # error propgagation
                    dfdn = calculate_error_propagation(dfdn_prev, M_prev, f_est, f_est_prev, self.n, self.eff)
                
                # covariance matrix
                cov = dfdn @ self.cov_n @ dfdn.T
                f_est_err = np.sqrt(np.diag(cov))
                
                # previous unfolding matrix
                M_prev = M
                dfdn_prev = dfdn
    
                # metric distance between consecutive estimates
                self.error = eval(f'{self.metric}(f_est, f_est_prev)')

                if self.epsilon > self.error and self.convergence_break == True:
                    if self.verbose == True:
                        print(f'error < epsilon, {iteration} Iterations.')
                        break
                        
                    else:
                        break
                
            self.f_est = f_est
            self.f_est_err = f_est_err
            self.cov = cov
            self.iteration = iteration
            
            return f_est, f_est_err
    
        else:
            print(f'Not fitted yet.')
            
@jit(nopython=True)
def calculate_error_propagation(dfdn_prev, M_prev, f_est, f_est_prev, n, eff):
    ''' Corrected error calculation by Adye, T.
    Uses error propagation matrices.
    '''
    
    
    n_bins_true = dfdn_prev.shape[0]
    n_bins_est = dfdn_prev.shape[1]
    
    dfdn = np.zeros((n_bins_true, n_bins_est))
    for i in range(n_bins_true):
        for j in range(n_bins_est):
            
            dfdn[i][j] = M_prev[i][j] + (f_est[i] / f_est_prev[i]) * dfdn_prev[i][j]
            
            kl_sum = 0
            for k in range(n_bins_est):
                for l in range(n_bins_true):
                    kl_sum += n[k] / f_est_prev[l] * M_prev[i][k] * M_prev[l][k] * dfdn_prev[l][j]
            
            dfdn[i][j] -= kl_sum
            
    
    return dfdn