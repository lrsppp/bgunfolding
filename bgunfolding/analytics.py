from astropy import units as u
import numpy as np
import pandas as pd
import gc

def create_sample_from_obstime(t_obs: u.hr, gammas_data, protons_data, gammas_weight, protons_weight, replace = True):
    """
    Creates a random sample containing gamma- and proton-events in natural ratio for a given oberservation time.
    
    Parameters
    ----------
    t_obs : float
        observation time
    
    gammas_data : pandas DataFrame-Object
        contains gamma events
        
    protons_data : pandas DataFrame-Object
        contains proton events
        
    Returns
    -------
    sample : pandas DataFrame-Object
    
    """
    
    # expected events for 1 hour scaled   
    n_gammas = int(np.sum(gammas_weight * t_obs))
    n_protons = int(np.sum(protons_weight * t_obs))
    
    # draw sample
    gammas_sample = gammas_data.sample(n = n_gammas,
                                       weights = gammas_weight, replace = replace)
    
    protons_sample = protons_data.sample(n = n_protons,
                                         weights = protons_weight, replace = replace)
        
    sample = pd.concat([gammas_sample, protons_sample])
    
    return sample


def calc_n_gamma_events():
    """
    Calculates expected gamma events for given observation time, ..... DOC
    
    Parameters
    ----------
    t_obs : float
        Observation time
        
    e_max : float
    e_min : flaot
    max_impact : float
    flux_normalization : float
    e_ref : float
    
    Return
    ------
    n : float
        expected amount of gamma events
    """
    
    area = np.pi * max_impact**2
    
    # divide equation into parts
    part1 = t_obs * area * flux_normalization * e_ref
    part2 = e_max**(spectral_index + 1) - e_min**(spectral_index + 1)
    
    nom = part1 * part2
    denom = (spectral_index + 1) * (e_ref**(spectral_index + 1))
    
    # decompose / cancel magnitudes (for e.g. hrs/seconds or GeV/TeV)
    n = (nom / denom).decompose()
     
    return n


def calc_density(x, bins_x, n_off = 1, cut_overflow = False):
    if cut_overflow:
        x = x[1:-1]
        bins_x = bins_x[1:-1]
        
    x_, _ = np.histogram(x, bins_x, weights = np.full(len(x), 1 / n_off))
    return x_


def calc_response_matrix(x, y, bins_x, bins_y, weights = None, cut_overflow = False, normalize = True):
    """
    Creates Response Matrix
    
    """

    if weights is None:
        A, _, _ = np.histogram2d(x, y, bins = [bins_x, bins_y])
        
    else:
        A, _, _ = np.histogram2d(x, y, bins = [bins_x, bins_y], weights = weights)
    
    if normalize == True:
        A = A / A.sum(axis = 0)
        
    if cut_overflow == True:
        A = A[1:-1,1:-1]

    return A

def create_sample(data, weight, y , replace = False):
    """
    Creates a sample from data based on weight. ??
    
    Parameters
    ----------

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

def create_sample_from_query(sample, query):
    """
    Parameters
    ----------
    Returns
    -------
    
    """
    if isinstance(query, list):
        query_sample = pd.concat([sample.query(i).copy() for i in query], sort = True)
    
    else:
        query_sample = sample.query(query).copy()
        
    return query_sample
