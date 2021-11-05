from bgunfolding import dataset_info
import h5py
from fact.analysis.statistics import calc_gamma_obstime, calc_proton_obstime, calc_weights_powerlaw, calc_weights_logparabola, calc_simulated_flux_normalization
from sklearn.model_selection import train_test_split
from astropy import units as u
from joblib import Parallel, delayed

# bgunfolding
from bgunfolding.utils import calc_log_bins, hist, create_sample
from bgunfolding.analytics import calc_response_matrix, calc_density, create_sample_from_query
import numpy as np
from fact.io import read_h5py, read_simulated_spectrum, write_data
import pandas as pd
import msgpack
import msgpack_numpy as m

import time
class Sampler():
    """
    Helper class to create a sample containing both signal and background. 
    """

    def __init__(self, 
                 fp = None,
                 n_samples = None, 
                 obstime = None, 
                 prediction_threshold = None,
                 theta_cut = None,
                 n_bins_true = None,
                 n_bins_est = None,
                 e_min_true = None, 
                 e_max_true = None, 
                 e_min_est = None, 
                 e_max_est = None,
                 test_size = 0.75):
        """
        Parameters
        ----------
        n_samples : int
        obstime : int astropy units
        prediction_threshold : float
            Gamma Prediction (selection cut parameter)
        theta_cut : float
            Distance to source position (selection cut parameter)
        n_bins_true : int
            Number of bins in true spectrum
        n_bins_est : int
            Number of bins in estimated spectrum
        e_min_true : int or float
        e_max_true : int or float
        e_min_est : int or float
        e_max_est : int or float
        test_size : float
            Size of test (gamma) sample 
        """
        self.fp = fp
        self.n_samples = n_samples
        self.obstime = obstime
        self.prediction_threshold = prediction_threshold
        self.theta_cut = theta_cut
        self.n_bins_true = n_bins_true
        self.n_bins_est = n_bins_est
        self.e_min_true = e_min_true
        self.e_max_true = e_max_true
        self.e_min_est = e_min_est
        self.e_max_est = e_max_est

        self.is_weighted = False
        self.is_fitted = False
        self.test_size = test_size
        self.train_size = 1 - test_size
        
        # define queries based on selection cut parameters
        self.on_query = f'gamma_prediction > {self.prediction_threshold} and theta_deg <= {self.theta_cut}'
        self.off_query = [f'gamma_prediction > {self.prediction_threshold} and theta_deg_off_{j} <= {self.theta_cut}' for j in range(1,6)]
        
        
    def read_data(self, 
                  gammas_train_fp,
                  gammas_test_fp,
                  gammas_corsika_fp, 
                  gammas_info, 
                  protons_fp, 
                  protons_corsika_fp, 
                  protons_info):
        """
        gammas_train : str
            Filename of gammas train dataset containing CERES for FACT response simulations. 
            Used to train response matrix.
        gammas_test : str
            Filename of gammas test dataset containing CERES for FACT response simulations.
            Used to create test samples.
        gammas_corsika : str
            Filename of gammas corsika dataset.
        gammas_info : dict
            Dictionary containing all information about the datasets. Also containing
            information about flux normalization constants used for reweighting spectra.
        protons : str
        protons_corsika : str
        protons_info : dict
        """
        # define column for ceres data
        events_columns = ['gamma_energy_prediction',
                         'gamma_prediction',
                         'theta_deg',
                         'corsika_event_header_event_number',
                         'corsika_event_header_total_energy',]
        
        events_columns_protons = events_columns + [f'theta_deg_off_{i+1}' for i in range(5)]
    
        # read corsika data
        gammas_corsika_total_energy = h5py.File(gammas_corsika_fp, mode = 'r')['corsika_events']['total_energy']
        # gammas_corsika_event_number = h5py.File(gammas_corsika_fp, mode = 'r')['corsika_events']['event_number']
        
        # read out spectrum info
        corsika_dict = {'gammas': gammas_corsika_fp, 'protons': protons_corsika_fp}
        try:
            for name, spectrum_fp in corsika_dict.items():
                for key, val in read_simulated_spectrum(spectrum_fp).items():
                    self.__dict__[f'{key}_{name}'] = val
                    
        except Exception:
            print('Could not read information from \"read_simulated_spectrum\"')
        
        # read out length of corsika file
        self.n_protons_corsika = len(h5py.File(protons_corsika_fp, mode = 'r')['corsika_events']['total_energy']) # only len necessary for reweighting
        self.n_gammas_corsika = len(gammas_corsika_total_energy)
        
        # read gammas and protons
        protons = read_h5py(protons_fp, key = 'events', columns = events_columns_protons)
        gammas_train = read_h5py(gammas_train_fp, key = 'events')
        gammas_test = read_h5py(gammas_test_fp, key = 'events')
        
        # store
        self.gammas_train = gammas_train
        self.gammas_test = gammas_test
        self.protons = protons
        
        # self.gammas_corsika_event_number = gammas_corsika_event_number
        self.gammas_corsika_total_energy = gammas_corsika_total_energy

        self.gammas_info = gammas_info
        self.protons_info = protons_info
        
        self.is_fitted = True
        
        self.gammas_train_fp = gammas_train_fp
        self.gammas_test_fp = gammas_test_fp
        self.gammas_corsika_fp = gammas_corsika_fp

        self.protons_fp = protons_fp
        self.protons_corsika_fp = protons_corsika_fp
        
        # calc binning
        self.calc_bins()
        
    def calc_bins(self):
        """
        Calculates equidistant log bins for estimated and true energy spectra.
        """
        
        if self.is_fitted:
            self.bins_est = calc_log_bins(e_min = self.e_min_est,
                                          e_max = self.e_max_est, over_under = True,
                                          n_bins = self.n_bins_est)

            self.bins_true = calc_log_bins(e_min = self.e_min_true,
                                           e_max = self.e_max_true, over_under = True,
                                           n_bins = self.n_bins_true)

            self.mids_true = 0.5 * (self.bins_true[1:] + self.bins_true[:-1])
            self.mids_est = 0.5 * (self.bins_est[1:] + self.bins_est[:-1]) 
            
        else:
            print('Sampler not fitted yet.')
        
    def calc_area_eff(self, delete = True):
        """
        Calculates effective area based on train dataset.
        """
        if self.is_fitted:
            n_simulated, _ = np.histogram(self.gammas_corsika_total_energy, bins = self.bins_true)
            
            # save memory
            if delete:
                del self.gammas_corsika_total_energy
            
            gammas_train_detected = self.gammas_train.query(self.on_query)
            n_detected, _ = np.histogram(gammas_train_detected['corsika_event_header_total_energy'], bins = self.bins_true)

            train_size = 1 - self.test_size
            acceptance = n_detected / (n_simulated * self.gammas_info['sample_fraction'] * train_size)
            area_sim = self.gammas_info['scatter_radius']**2 * np.pi
            area_eff = acceptance * area_sim.value
            
            # store
            self.train_size = train_size
            self.acceptance = np.array([acceptance for i in range(self.n_samples)])
            self.area_sim = np.array([area_sim.value for i in range(self.n_samples)]) # only value cause numpy cannot handle astropy units
            self.area_eff = np.array([area_eff for i in range(self.n_samples)])

            self.n_simulated = np.array([n_simulated for i in range(self.n_samples)])
            self.n_detected = np.array([n_detected for i in range(self.n_samples)])
             
        else:
            print('Sampler not fitted yet.')
            
    def calc_weights(self, gamma_spectrum_type):
        """
        Parameters
        ----------
        gamma_spectrum_type : str
            Choose between logparabola (https://arxiv.org/pdf/1406.6892.pdf) and
            powerlaw spectrum (Aharonian, F. et al. (HEGRA collaboration) 2004, ApJ 614, 897)
        """
        self.gamma_spectrum_type = gamma_spectrum_type
        
        for dataset in ['test', 'train']:
            if gamma_spectrum_type == 'powerlaw':
                self.__dict__[f'gammas_{dataset}']['weight'] = calc_weights_powerlaw(energy = self.__dict__[f'gammas_{dataset}']['corsika_event_header_total_energy'].values * u.GeV,
                                                                  obstime = self.obstime,
                                                                  n_events = self.n_gammas_corsika * self.__dict__[f'{dataset}_size'],
                                                                  e_min = self.gammas_info['energy_min'],
                                                                  e_max = self.gammas_info['energy_max'],
                                                                  simulated_index = self.gammas_info['simulated_index'],
                                                                  scatter_radius = self.gammas_info['scatter_radius'],
                                                                  target_index = self.gammas_info['target_hegra'],
                                                                  flux_normalization = self.gammas_info['flux_normalization_hegra_powerlaw'],
                                                                  e_ref = self.gammas_info['energy_ref'],
                                                                  sample_fraction = self.gammas_info['sample_fraction'])

            elif gamma_spectrum_type == 'logparabola':
                self.__dict__[f'gammas_{dataset}']['weight'] = calc_weights_logparabola(energy = self.__dict__[f'gammas_{dataset}']['corsika_event_header_total_energy'].values * u.GeV,
                                                                 obstime = self.obstime,
                                                                 n_events = self.n_gammas_corsika * self.__dict__[f'{dataset}_size'],
                                                                 e_min = self.gammas_info['energy_min'],
                                                                 e_max = self.gammas_info['energy_max'],
                                                                 simulated_index = self.gammas_info['simulated_index'],
                                                                 scatter_radius = self.gammas_info['scatter_radius'],
                                                                 target_a = self.gammas_info['target_magic_a'],
                                                                 target_b = self.gammas_info['target_magic_b'],
                                                                 flux_normalization = self.gammas_info['flux_normalization_magic_logparabola'],
                                                                 e_ref = self.gammas_info['energy_ref'],
                                                                 sample_fraction = self.gammas_info['sample_fraction'])
        
        # protons weights
        self.protons['weight'] = calc_weights_powerlaw(energy = self.protons['corsika_event_header_total_energy'].values * u.GeV,
                                    obstime = self.obstime,
                                    n_events = self.n_protons_corsika * self.protons_info['reuse_factor'],
                                    e_min = self.protons_info['energy_min'],
                                    e_max = self.protons_info['energy_max'],
                                    simulated_index = self.protons_info['simulated_index'],
                                    scatter_radius = self.protons_info['scatter_radius'],
                                    target_index = self.protons_info['target_index'],
                                    flux_normalization = self.protons_info['flux_normalization_pdg'],
                                    e_ref = self.protons_info['energy_ref'],
                                    sample_fraction = self.protons_info['sample_fraction'],
                                    viewcone = self.protons_info['viewcone'])
        
        self.n_gammas_test = int(np.sum(self.gammas_test['weight']))
        self.n_protons = int(np.sum(self.protons['weight']))
        
        self.is_weighted = True
    
    
    def calc_response_matrix(self):
        """
        Calculates repsonse matrix based on training dataset.
        Each element contains information about causes smearing into effects bins.
        
        Parameters
        ----------
        train_weighted : 
        """
        gammas_train_detected = self.gammas_train.query(self.on_query)
        # gammas_train_detected['weight'] 
        
        # response matrix
        A = calc_response_matrix(gammas_train_detected['gamma_energy_prediction'],
                                 gammas_train_detected['corsika_event_header_total_energy'],
                                 self.bins_est,
                                 self.bins_true,
                                 cut_overflow = False,
                                 normalize = False)
        
        # calc efficiencies
        eff = np.zeros(self.n_bins_true + 2)

        H = A / A.sum(axis = 0)
        for i in range(self.n_bins_true + 2):
            eff[i] = 1 - H[:,i][0] - H[:,i][-1]

        # f train
        f_train, _ = np.histogram(gammas_train_detected['corsika_event_header_total_energy'], 
                                  self.bins_true, weights = gammas_train_detected['weight'])
        
        self.f_train = np.array([f_train for i in range(self.n_samples)])
        self.A = np.array([A for i in range(self.n_samples)])
        self.eff = np.array([eff for i in range(self.n_samples)])
        
        
    def init(self, spectrum_type):
        """
        Helper function to initialize the sampler. Calling different functions.
        Especially calculating bins and effective area. The repsonse matrix is also generated from
        training dataset.
        
        Parameters
        ----------
        spectrum_type : str
            Choose weighting of simulated data (reweighting to 'logparabola' or 'powerlaw')
        """
        
        print('Running methods \"calc_weights\", \"calc_bins\", \"calc_area_eff\", \"calc_response_matrix\"')
        self.calc_weights(spectrum_type)
        self.calc_bins()
        self.calc_area_eff()
        self.calc_response_matrix()

    def create_samples(self, n_cores = 2, verbose = True):
        """
        Helper function. Creates the i-th sample of n_samples.
        """
        
        start = time.time()
        
        self.g = np.zeros((self.n_samples, self.n_bins_est + 2))
        self.b = np.zeros((self.n_samples, self.n_bins_est + 2))
        self.f = np.zeros((self.n_samples, self.n_bins_true + 2))
        
        # sum weights
        n_gammas_sample = np.sum(self.gammas_test.weight)
        n_protons_sample = np.sum(self.protons.weight)

        # idx
        gammas_idx = self.gammas_test.index.values
        protons_idx = self.protons.index.values

        # sampling weights
        p_gammas = self.gammas_test.weight.values / n_gammas_sample
        p_protons = self.protons.weight.values / n_protons_sample

        # parallelize idx sampling
        print('start creating idx')
        replace = False # sampling without replacement
        gammas_idx_list = Parallel(n_jobs = n_cores, backend = 'loky')(delayed(sample_idx)(gammas_idx, int(n_gammas_sample), replace, p_gammas) for i in range(self.n_samples))
        protons_idx_list = Parallel(n_jobs = n_cores, backend = 'loky', verbose = verbose)(delayed(sample_idx)(protons_idx, int(n_protons_sample), replace, p_protons) for i in range(self.n_samples))

        # loop through idxs
        print('start creating fgb from idx')
        for i in range(self.n_samples):
            
            if i % 50 == 0:
                print(f'{i} / {self.n_samples}')
            
            gammas_sample = self.gammas_test.loc[gammas_idx_list[i]]
            protons_sample = self.protons.loc[protons_idx_list[i]]

            g = np.histogram(gammas_sample.query(self.on_query)['gamma_energy_prediction'], bins = self.bins_est)[0]
            g += np.histogram(protons_sample.query(self.on_query)['gamma_energy_prediction'], bins = self.bins_est)[0]

            n_off = self.gammas_info['n_off']
            b = np.zeros(self.n_bins_est + 2)
            for j in range(n_off):
                b += np.histogram(gammas_sample.query(self.off_query[j])['gamma_energy_prediction'], bins = self.bins_est)[0]
                b += np.histogram(protons_sample.query(self.off_query[j])['gamma_energy_prediction'], bins = self.bins_est)[0]

            # consideration of n_off off-regions
            b = b / n_off
            
            # true density
            f = np.histogram(gammas_sample.query(self.on_query)['corsika_event_header_total_energy'], bins = self.bins_true)[0]

            # store
            self.f[i] = f
            self.g[i] = g
            self.b[i] = b
            
        # end timer
        end = time.time() - start
        print(f'Created {self.n_samples} samples in {np.round(end, 2)} seconds.')
            
    def save_sample(self, fp = None):
        """
        Save sample as binary.
        
        Parameters
        ----------
        fp : string
            Filepath
        """
        
        if fp == None:
            fp = self.fp
        
        # save every possible key / value of sampler object
        dataDict = dict()
        storable_dtypes = (str, int, float, bool, np.float32, np.float64, list, np.ndarray)
        
        for key, val in self.__dict__.items():
            if isinstance(val, storable_dtypes) == True:
                dataDict[key] = val
        
        m.patch()
        try:
            binary = msgpack.packb(dataDict, use_bin_type  = True)
            with open(fp, 'wb') as f:
                f.write(binary)
        except Exception as e:
            print(e)
            
    def read_sample(self, fp):
            """
            Load a sample saved as binary.

            Parameters
            ----------
            fp : string
                Filepath
            """
            m.patch()
            with open(fp, 'rb') as f:
                rec = msgpack.unpackb(f.read(), encoding = 'utf-8')

            for key, val in rec.items():
                self.__dict__[key] = val

            print(f'Data succesfully read. (n_samples : {self.n_samples})')
        
    def read_sample_config(self, config_dict, square_theta = True, add_units = True):
        """
        Parameters
        ----------
        config_dict : dictionary
        """
        
        for key, item in config_dict.items():
            self.__dict__[key] = item
            
        self.n_samples = int(self.n_samples)
        self.n_bins_true = int(self.n_bins_true)
        self.n_bins_est = int(self.n_bins_est)   
        
        if add_units:
            self.obstime = self.obstime * u.hr
        
        if square_theta:
            self.theta_cut = np.sqrt(self.theta_cut)
            
        # define queries based on selection cut parameters
        if self.theta_cut == 0:
            self.on_query = f'gamma_prediction > {self.prediction_threshold}'
            self.off_query = [f'gamma_prediction > {self.prediction_threshold}' for j in range(1,6)]
        elif self.prediction_threshold == 0:
            self.on_query = f'theta_deg <= {self.theta_cut}'
            self.off_query = [f'theta_deg_off_{j} <= {self.theta_cut}' for j in range(1,6)]
        else:
            self.on_query = f'gamma_prediction > {self.prediction_threshold} and theta_deg <= {self.theta_cut}'
            self.off_query = [f'gamma_prediction > {self.prediction_threshold} and theta_deg_off_{j} <= {self.theta_cut}' for j in range(1,6)]
            
    def load_one(self, i):
        ''' Helper function to load one sample. '''
        return self.f[i], self.g[i], self.b[i], self.A[i], self.area_eff[i], self.acceptance[i], self.eff[i]


        
def sample_idx(idx, size, replace, p):
    return np.random.choice(idx, size = size, replace = replace, p = p)


def gammas_train_test_split(gammas_fp, test_size = 0.75, random_state = 2020, write_to_files = False, gammas_train_fp = None, gammas_test_fp = None):
    gammas = read_h5py(gammas_fp, key = 'events')
    gammas_train, gammas_test = train_test_split(gammas, test_size = test_size, random_state = random_state)
    
    if write_to_files:
        if gammas_train_fp != None:
            write_data(gammas_train, file_path = gammas_train_fp, key = 'events', mode = 'w')
            print(f'Succesfully created training data {gammas_train_fp}')
        if gammas_test_fp != None:
            write_data(gammas_test, file_path = gammas_test_fp, key = 'events', mode = 'w')
            print(f'Succesfully created test data {gammas_test_fp}')