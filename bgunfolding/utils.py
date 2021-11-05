import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import msgpack
import msgpack_numpy as m
from os.path import exists, split, join
from os import makedirs

def quick_savefig(fign, config):
    
    fp = join('/home/lars/thesis-bgunfolding/Plots', config['sample']['filename'])
    print(fp)
    
    if not exists(fp):
        makedirs(fp)
        
    plt.savefig(join(fp, fign))
    
class Results():
    """
    Helper object to save deconvolution results from different samples. 
    These can then be conveniently stored using `msgpack` (and `msgpack_numpy`) and used for later analysis.
    """
    def __init__(self):
        self.level = 0
        self.level_names = ['level', 'level_names']
        
    def add(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, str(key)) == True:
                x = self.__dict__[key]
                if type(x) != list:
                    self.__dict__[key] = [x] + [val]
                else:
                    x.append(val)
                    self.__dict__[key] = x

            else:
                self.__dict__.update({key: val})
                
    def add_dict(self, d):
        for key, val in d.items():
            if hasattr(self, str(key)) == True:
                x = self.__dict__[key]
                if type(x) != list:
                    self.__dict__[key] = [x] + [val]
                else:
                    x.append(val)
                    self.__dict__[key] = x

            else:
                self.__dict__.update({key: val})
            
    def add_level(self, name):
        self.level_names = self.level_names + [name]
        
        d = {}
        for key, val in self.__dict__.items():
            if key not in self.level_names:
                d.update({key: val})
                
        self.__dict__[name] = d
        
        k = list(self.__dict__.keys())
        for key in k:
            if key not in self.level_names:
                self.__dict__.pop(key)
                
        self.level += 1
                
    def write(self, fp, overwrite = False):
        m.patch()
        p, f = split(fp)
        if not exists(p):
            makedirs(p, exist_ok = False)
            
        if not exists(fp):
            print(f'file saved {fp}')
            binary = msgpack.packb(self.__dict__, use_bin_type  = True)
            with open(fp, 'wb') as file:
                file.write(binary)

        elif overwrite:
            print(f'file overwritten {fp}')
            binary = msgpack.packb(self.__dict__, use_bin_type  = True)
            with open(fp, 'wb') as file:
                file.write(binary)
                
        else:
            print(f'file already exists {fp}')
            
    def read(self, fp):
        m.patch()
        with open(fp, 'rb') as file:
            rec = msgpack.unpackb(file.read(), encoding = 'utf-8')
        self.__dict__ = rec
        
    def reset(self):
        self.__dict__ = {}
        self.level = 0
        self.level_names = ['level', 'level_names']
        
    def dictify(self):

        r = self.__dict__
        for key, val in r.items():
            r[key] = np.array(val)

        return r

def create_sample_from_query(sample, query):
    """
    Parameters
    ----------
    Returns
    -------
    
    """
    if isinstance(query, list):
        query_sample = pd.concat([sample.query(i).copy() for i in query])
    
    else:
        query_sample = sample.query(query).copy()
        
    return query_sample


def create_sample(data, weight, y , replace = False):
    """
    Creates a sample from data based on weight. ??
    
    Parameters
    ----------
    data : 
    y : int
        class variable
    Returns
    -------
    
    """
    weight_sum = np.sum(weight)
    size = np.random.poisson(lam = weight_sum)
    
    indices = np.random.choice(data.index, size = size,
                               p = weight / weight_sum, replace = replace)

    sample = data.loc[indices].copy()
    sample['y'] = y
    
    return sample

def calc_log_bins(e_min, e_max, n_bins, over_under = False, under = 10, over = 1e5):
    bins = np.logspace(np.log10(e_min), np.log10(e_max), n_bins + 1)
    if over_under:
        bins = np.concatenate([[under], bins, [over]])
    return bins
    
    
def cov2corr(cov, return_std = False):
    '''convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    '''
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr
    
    
def hist(ax,
         binned_data, 
         bins, 
         label = '',
         color = None,
         linewidth = 1.5,
         cut_overflow = True, 
         density = False):
    
        """
        Plot Histogram of binned data, for e.g. estimated density f.
        """
        mids = (bins[1:] + bins[:-1]) * 0.5
        
        if cut_overflow == True:
            bins = bins[1:-1]
            mids = mids[1:-1]
            binned_data = binned_data[1:-1]
        
        else:
            binned_data = binned_data
            
        ax.hist(mids, 
                 bins, 
                 weights = binned_data, 
                 histtype = 'step', 
                 label = label, 
                 density = density,
                 color = color,
                 linewidth = linewidth)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc = 'best')
        
def hist_error(ax, binned_data, yerr, bins, color = None, alpha = 0.2, cut_overflow = True):
    """
    Plot error for binned data using matplotlib's fill_between
    """
    err_low = binned_data + yerr
    err_high = binned_data - yerr
    
    if cut_overflow:
        err_low = err_low[1:-1]
        err_high = err_high[1:-1]
        binned_data = binned_data[1:-1]
        bins = bins[1:-1]
    
    for i in range(len(bins) - 1):
        x_low = bins[i]
        x_high = bins[i+1]

        x = np.linspace(x_low, x_high, 150)
    
        ax.fill_between(x,
                         y1 = err_low[i],
                         y2 = err_high[i],
                         color = color,
                         alpha = alpha)    
        
def calc_glob_cc(cov):
    glob_cc = np.mean(np.sqrt(1 - 1 / ( np.diag(cov) * np.diag(np.linalg.inv(cov)))))

    return glob_cc

def estimate_minimum(tau_space, glob_cc, return_index = False):
    try:
        min_index = np.where(glob_cc == np.min(glob_cc))[0][0]
        tau_est = tau_space[min_index]
        
        if return_index:
            return tau_est, min_index
        
        else:
            return tau_est
        
    except:
        print('Could not estimate regularization parameter.')
        
def unfolding_multi_plot(bins_true, mids_true, f_est, f_est_err, f_true, label_est, label_true = 'True', subplot_shape = [2, 2], cut_overflow = True, legend_fs = 12, title_fs = 12):

    fig, ax = plt.subplots(subplot_shape[0], subplot_shape[1])
    axes = ax.flatten()
    
    if cut_overflow:
        bins_true = bins_true[1:-1]
        mids_true = mids_true[1:-1]

    for i, axis in enumerate(axes):

        axis.hist(mids_true, 
                  bins_true, 
                  weights = f_true[i][1:-1], 
                  label = label_true,
                  histtype = 'step',
                  color = 'C1',
                  linewidth = 2)
        axis.hist(mids_true, 
                  bins_true, 
                  weights = f_est[i][1:-1], 
                  label = label_est,
                  histtype = 'step',
                  color = 'C0',
                  linewidth = 2)

        err_low = f_est[i][1:-1] - f_est_err[i][1:-1]
        err_high = f_est[i][1:-1] + f_est_err[i][1:-1]

        for j in range(len(bins_true) - 1):
            x_low = bins_true[j]
            x_high = bins_true[j+1]

            x = np.linspace(x_low, x_high, 150)

            axis.fill_between(x,
                             y1 = err_low[j],
                             y2 = err_high[j],
                             alpha = 0.2,
                             color = 'C0')    


        axis.set_title(f'Sample {i+1}', fontsize = title_fs)
        axis.set_xlabel(r'$\mathrm{Energy}\,E\,/\,\mathrm{GeV}$')
        axis.set_ylabel(r'$\mathrm{Counts}$')
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.legend(loc = 'best', fontsize = legend_fs)

    plt.tight_layout()
    
def read_mcmc_data(fp):
    m.patch()
    with open(fp, 'rb') as f:
        rec = msgpack.unpackb(f.read(), encoding = 'utf-8')

    data = rec['data']
    meta = rec['meta']

    return data, meta


def MAD(x, scale = 1):
    """
    Estimator for Median Absolute Deviation
    
    scale : float
        1.4826 for consistency with normal distributed data.
    """
    mad = np.median(np.abs(x - np.median(x)))
    
    return scale * mad


def calc_med_corr(d, n, a = 1, b = 1):
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = d[i] - np.median(d[i])
            y = d[j] - np.median(d[j])

            corr[i][j] = np.median((x * b / MAD(d[i], a)) * (y * b / MAD(d[j], a)))
        
    return corr


def calc_med_cov(d, n):
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = d[i] - np.median(d[i])
            y = d[j] - np.median(d[j])

            cov[i][j] = np.median(x * y)
        
    return cov