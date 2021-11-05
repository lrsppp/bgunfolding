import numpy as np
from numba import jit

def log_prior(f):
    if (f < 0).any():
        return -np.inf
    
    return 0

def llh_poisson(f, g, b, A, eps = 1e-12, cut_overflow = True):
    """
    Poisson Likelihood
    """
    
    if cut_overflow:
        f = f[1:-1]
        g = g[1:-1]
        b = b[1:-1]
        A = A[1:-1,1:-1]
    
    return np.sum(g * np.log(A @ f + b + eps) - (A @ f + b))

def llh_tikhonov(f, C, tau, acceptance, cut_overflow = True):
    """
    Likelihood Regularization Term
    """
    
    if cut_overflow:
        f = f[1:-1]
        acceptance = acceptance[1:-1]
    
    f = np.log10(f / acceptance)
    
    CT_C = tau * C.T @ C
    
    return -0.5 * f.T @ CT_C @ f

def hess_poisson(f, g, b, A, eps = 1e-12, cut_overflow = True):
    if cut_overflow:
        f = f[1:-1]
        g = g[1:-1]
        b = b[1:-1]
        A = A[1:-1,1:-1]   
    
    hess = np.dot(A.T, np.dot(np.diag(g / (A @ f + b + eps) ** 2), A))
    
    return hess

@jit
def hess_tikhonov(f, C, tau, acceptance, cut_overflow = True):
    CT_C = C.T @ C
    
    # drop over- and underflow
    if cut_overflow:
        f = f[1:-1]
        acceptance = acceptance[1:-1]
    
    hess = np.zeros((len(f), len(f)))
    
    for k in range(len(f)):
        for l in range(len(f)):
            
            if k != l:
                hess[k][l] = -0.5 * tau * (CT_C[l][k] + CT_C[k][l]) / (np.log(10)**2 * f[k] * f[l])
                
            else:
                sn = 0
                for i in range(len(f)):
                    if i == k:
                        sn += (CT_C[i][k] + CT_C[k][i]) * (1 - np.log(f[i] / acceptance[i]))
                    else:
                        sn += (CT_C[i][k] + CT_C[k][i]) * (-np.log(f[i] / acceptance[i]))
                        
                hess[k][l] = -0.5 * tau * sn / (np.log(10)**2 * f[k]**2)
    
    return -hess
