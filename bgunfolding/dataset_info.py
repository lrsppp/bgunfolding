from astropy import units as u

'''
The gamma ray energy spectrum of the Crab Nebula as measured by the HEGRA experiment.
See Aharonian, F. et al. (HEGRA collaboration) 2004, ApJ 614, 897
'''

'''
See MAGIC paper
https://arxiv.org/pdf/1406.6892.pdf
'''

gammas_info = {'energy_max': 100000 * u.GeV,
               'energy_min': 200 * u.GeV,
               'simulated_index': -2,
               'scatter_radius': 300 * u.m,
               'energy_ref': 1 * u.TeV,
               'flux_normalization_hegra_powerlaw': 2.83e-14 / (u.GeV * u.s * u.cm**2),
               'target_hegra': -2.62,
               'flux_normalization_magic_logparabola': 3.23e-11 / (u.TeV * u.s * u.cm**2),
               'target_magic_a': -2.47,
               'target_magic_b': -0.24,
               'n_events': 54000000,
               'sample_fraction': 0.75,
               'n_off': 5}

protons_info = {'energy_max': 200000 * u.GeV,
                'energy_min': 100 * u.GeV,
                'simulated_index': -2.0,
                'target_index': -2.7,
                'scatter_radius': 500 * u.m,
                'viewcone': 6 * u.deg,
                'energy_ref': 1 * u.GeV,
                'flux_normalization_pdg': 1.8e4 / (u.sr * u.s * u.m**2 * u.GeV),
                'flux_normalization_bess': 9.6e-9 / (u.GeV * u.s * u.cm**2 * u.sr),
                'n_showers': 216000000,
                'reuse_factor': 20,
                'sample_fraction': 0.75}
