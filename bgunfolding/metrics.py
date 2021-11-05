import numpy as np
from scipy.stats import kstwobign, wasserstein_distance

__all__ = ['_normalize', 
           'chisq_sym',
           'chisq_asym',
           'kl',
           'emd',
           'pull_statistic']


def _normalize(f_est, f_true):
    """
    Helper function to normalize estimated density f_est and true density f_true.
    
        
    Parameters
    -----------
    f_est : array
        Binned data of estimated density f_est.
    f_true : array
        Binned data of true density f_true.
        
    Returns
    -------
    f_est_norm : array
        Normalized binned data of estimated density f_est.
    f_true_norm : array
        Normalized binned data of true density f_true.
        
    """
    n_dim = f_est.ndim
    
    if n_dim == 1:
        f_est_norm = f_est / np.sum(f_est)
        f_true_norm = f_true / np.sum(f_true)

    else:
        f_est_norm = f_est / f_est.sum(axis = 1).reshape(len(f_est), 1)
        f_true_norm = f_true / f_true.sum(axis = 1).reshape(len(f_est), 1)
    
    return f_est_norm, f_true_norm


def chisq_sym(f_est, f_true, cut_overflow = True, replace_zeros = True, zero = 1e-9):
    """
    Symmetric Chi-Square Distance

    Parameters
    -----------
    f_est : array
        Binned data of estimated density f_est.
    f_true : array
        Binned data of true density f_true.
        
    Returns
    -------
    dist : float
        Distance between two densities
    """
    if cut_overflow:
        f_est = f_est[1:-1]
        f_true = f_true[1:-1]
        
    f_est, f_true = _normalize(f_est, f_true)
    n_dim = f_est.ndim   
    # replace zeros with small number
    if replace_zeros:
        f_est[f_est == 0] = zero
        f_true[f_true == 0] = zero

    if n_dim == 1:
        return 2 * np.sum((f_est - f_true)**2 / (f_est + f_true))
    
    else:
        return 2 * np.sum((f_est - f_true)**2 / (f_est + f_true), axis = 1)


def chisq_asym(f_est, f_true, cut_overflow = True, replace_zeros = True, zero = 1e-9):
    """
    Asymmetric Chi-Square Distance.
    
    Parameters
    -----------
    f_est : array
        Binned data of estimated density f_est.
    f_true : array
        Binned data of true density f_true.
        
    Returns
    -------
    dist : float
        Distance between two densities
    """
    if cut_overflow:
        f_est = f_est[1:-1]
        f_true = f_true[1:-1]
        
    f_est, f_true = _normalize(f_est, f_true)
    n_dim = f_est.ndim
    
    # replace zeros with small number
    if replace_zeros:
        f_est[f_est == 0] = zero
        f_true[f_true == 0] = zero
        
    if n_dim == 1:
        return np.sum((f_est - f_true)**2 / f_true)
    
    else:
        return np.sum((f_est - f_true)**2 / f_true, axis = 1)

def kl(f_est, f_true, cut_overflow = True, replace_zeros = True, zero = 1e-9):
    """
    Kullback-Leibler Distance
    
    Parameters
    -----------
    f_est : array
        Binned data of estimated density f_est.
    f_true : array
        Binned data of true density f_true.
        
    Returns
    -------
    dist : float
        Distance between two densities
    """

    if cut_overflow:
        f_est = f_est[1:-1]
        f_true = f_true[1:-1]
        
    f_est, f_true = _normalize(f_est, f_true)
    
    # replace zeros with small number
    if replace_zeros:
        f_est[f_est == 0] = zero
        f_true[f_true == 0] = zero
        
    n_dim = f_est.ndim

    if n_dim == 1:
        return np.sum(f_est * np.log(f_est / f_true))
    
    else:
        return np.sum(f_est * np.log(f_est / f_true), axis = 1)
    
def emd(f_est, f_true, cut_overflow = True, v = None, u = None):
    """
    Earth Mover's Distance (also known as Wasserstein-Distance).
    
    Parameters
    -----------
    f_est : array
        Binned data of estimated density f_est.
    f_true : array
        Binned data of true density f_true.
        
    Returns
    -------
    dist : float
        Distance between two densities
    """
    if cut_overflow:
        f_est = f_est[1:-1]
        f_true = f_true[1:-1]
    
    f_est, f_true = _normalize(f_est, f_true)
    n_dim = f_est.ndim
    
    if u == None:
        u = np.arange(len(f_est))
    if v == None:
        v = np.arange(len(f_true))
        
    if n_dim == 1:
        return wasserstein_distance(u, 
                                    v, 
                                    f_est, 
                                    f_true)
    
    else:
        dist = np.zeros(len(f_est))
        for i in range(len(f_est)):
            dist[i] = wasserstein_distance(np.arange(len(f_est[i])), 
                                           np.arange(len(f_true[i])), 
                                           f_est[i], 
                                           f_true[i])
        return dist


def pull_statistic(f_est, f_est_err, f_true, cut_overflow = True):
    """
    Pull Statistic.
    
    Used to compare unfolding results from different pulls and reveal wether the results are
    too varied or if they are biased in any way.
    """
    
    if cut_overflow:
        f_est = f_est[1:-1]
        f_true = f_true[1:-1]
        f_est_err = f_est_err[1:-1]

    return (np.array(f_est) - np.array(f_true)) / np.array(f_est_err)  
    

def pull_statistic_percentiles(est_50, est_84, est_16, est_true, cut_overflow = True):
    if cut_overflow:
        est_50 = est_50[1:-1]
        est_84 = est_84[1:-1]
        est_16 = est_16[1:-1]
        est_true = est_true[1:-1]
        
    return np.array((est_50 - est_true) / (0.5 * (est_84 - est_16)))
