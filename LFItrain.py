import numpy as np
import matplotlib.pyplot as plt
import pydelfi.priors as priors
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import fact.io
import pandas as pd

# Get training data.
input_file = 'gammas-diffuse.hdf5'

df_arr = fact.io.read_data(input_file, key='array_events')
df_tel = fact.io.read_data(input_file, key='telescope_events')

df = pd.merge(df_tel, df_arr, 
            on=['array_event_id', 'run_id'])

##################

# Get testing data.
test_file = 'gammas-1deg.hdf5'

test_arr = fact.io.read_data(test_file, key='array_events')
test_tel = fact.io.read_data(test_file, key='telescope_events')

test = pd.merge(test_tel, test_arr, 
            on=['array_event_id', 'run_id'])

# Extract values.
features = ['concentration_1', 'intensity', 
            'kurtosis', 'length',
            'n_survived_pixels', 
            'skewness', 'tgradient', 
            'width']

theta = ['mc_energy', 'mc_alt', 'mc_az', 'mc_core_x', 'mc_core_y']

df = df[features+theta].dropna()
test = test[features+theta].dropna()

df['mc_az'][df['mc_az']<np.pi] = df['mc_az'][df['mc_az']<np.pi] + 2*np.pi
test['mc_az'][test['mc_az']<np.pi] = test['mc_az'][test['mc_az']<np.pi] + 2*np.pi

df['mc_energy'] = np.log10(df['mc_energy'])
test['mc_energy'] = np.log10(test['mc_energy'])

sim_compressed_data = np.array(df[features])
sim_params = np.array(df[theta]).astype(None)

i = 0
compressed_data = np.array(test[features])[i]
params = np.array(test[theta])[i:i+1]

# Initilize LFI

#Finv = np.genfromtxt('simulators/cosmic_shear/pre_ran_sims/Finv.dat')

#theta_fiducial = np.array([0.3, 0.8, 0.05, 0.70, 0.96])

lower = np.array([0, 1.05, 5.8, -400, -400])
upper = np.array([2.5, 1.4, 6.8, 400, 400])
prior = priors.Uniform(lower, upper)

NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=5, n_data=8, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=0),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=1, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=2, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=3, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=4, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=5, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=5)]

DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, 
                            #Finv = Finv, 
                            #theta_fiducial = theta_fiducial, 
                            param_limits = [lower, upper],
                            param_names = theta, 
                            # results_dir = '',
                            #input_normalization="fisher"
                           )

DelfiEnsemble.load_simulations(sim_compressed_data, sim_params)

# Train LFI

# DelfiEnsemble.fisher_pretraining(n_batch=5000, batch_size=100)

DelfiEnsemble.train_ndes(
    batch_size=50,
    validation_split=0.1,
    epochs=300,
    patience=20)

