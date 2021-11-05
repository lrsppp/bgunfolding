import numpy as np

class UnfoldingBase():
    """
    Base class.
    """
    def __init__(self, *args):
        self.is_fitted = False

    def fit(self, f, g, b, A, area_eff = None, acceptance = None, eff = None, normalize_response = True):
        """
        f : array-like
        g : array-like
        b : array-like
        A : array-like
        area_eff : array-like
        eff : array-like
            Efficiency e_i that cause i has an effect
        """
        self.is_fitted = True
        
        self.f = f
        self.g = g
        self.b = b
        self.H = A # unnormalized response matrix

        self.area_eff = area_eff
        self.acceptance = acceptance
        self.eff = eff
        
        # normalization
        if normalize_response == False:
            self.A = self.H
        else: 
            self.A = self.H / self.H.sum(axis = 0)
            
        self.n_bins_true = len(f)
        self.n_bins_est = len(g)
        self.cov_g = np.diag(np.sqrt(g))
        self.cov_b = np.diag(np.sqrt(b))
        
        self.n = self.g - self.b
        self.cov_n = self.cov_g + self.cov_b
