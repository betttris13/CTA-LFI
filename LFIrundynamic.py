import numpy as np
import matplotlib.pyplot as plt
import pydelfi.priors as priors
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import tensorflow as tf
import dynesty
from dynesty import utils as dyfunc
import corner
import fact.io
import pandas as pd
# tf.logging.set_verbosity(tf.logging.ERROR)

# Load test file and extract values.
test_file = 'gammas-1deg.hdf5'
# test_file = 'gammas-diffuse.hdf5'

test_arr = fact.io.read_data(test_file, key='array_events')
test_tel = fact.io.read_data(test_file, key='telescope_events')

test = pd.merge(test_tel, test_arr, 
            on=['array_event_id', 'run_id'])

features = ['concentration_1', 'intensity', 
            'kurtosis', 'length',
            'n_survived_pixels', 
            'skewness', 'tgradient', 
            'width']

theta = ['mc_energy', 'mc_alt', 'mc_az', 'mc_core_x', 'mc_core_y']

test = test[features+theta].dropna()
test['mc_az'][test['mc_az']<np.pi] = test['mc_az'][test['mc_az']<np.pi] + 2*np.pi
test['mc_energy'] = np.log10(test['mc_energy'])

# Setup Neural Density Estimator network
NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=5, n_data=8, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=0),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=1, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=2, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=3, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=4, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4),
            ndes.MixtureDensityNetwork(n_parameters=5, n_data=8, n_components=5, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=5)]

# Setup lower and upper value for each parameter and define LFI prior
lower = np.array([0, 1.05, 5.8, -400, -400])
upper = np.array([2.5, 1.4, 6.8, 400, 400])
pRange = upper - lower
prior = priors.Uniform(lower, upper)

# Initialize pydelfi with first image in dataset
compressed_data = np.array(test[features])[0]
DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, 
                            #Finv = Finv, 
                            #theta_fiducial = theta_fiducial, 
                            param_limits = [lower, upper],
                            param_names = theta,
                            restore = True,
                            #input_normalization="fisher"
                        )

# Index of first and last (inclusive) image to be run. I.e. run over range [first,last].
first = 0
last = 0

# Run recostruction over selected range.
for i in range(first,last+1):
    compressed_data = np.array(test[features])[i]
    params = np.array(test[theta])[i:i+1]
    DelfiEnsemble.data = compressed_data

    # Define posterior and prior for use by sampler. Prior should match LFI prior.
    posterior = lambda x: DelfiEnsemble.log_posterior_stacked(x, DelfiEnsemble.data)[0][0]

    def pr(u):
        return pRange*u+lower

    # Run sampler and extract samples and statistical values.
    sampler = dynesty.DynamicNestedSampler(posterior, pr, 5)
    sampler.run_nested()
    posterior_samples = sampler.results.samples

    mean = np.mean(posterior_samples, axis = 0)
    median = np.median(posterior_samples, axis = 0)
    v = np.argmax(sampler.results.logl) # Unssure if this is working. Almost certian there is a better way.
    mode = sampler.results.samples[v]

    # print(mean)
    # print(median)
    # print(mode)
    # print(cov)

    fig = corner.corner(posterior_samples, labels=theta)

    ndim = 5

    # Extract the axes
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for j in range(ndim):
        ax = axes[j, j]
        ax.axvline(params[0][j], color="g")
        # ax.axvline(mean[j], color="r")
        # ax.axvline(median[j], color="b")
        # ax.axvline(mode[j], color="y")
        
    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(params[0][xi], color="g")
            ax.axhline(params[0][yi], color="g")
            ax.plot(params[0][xi], params[0][yi], "sg")
            # ax.axvline(mean[xi], color="r")
            # ax.axhline(mean[yi], color="r")
            # ax.plot(mean[xi], mean[yi], "sr")
            # ax.axvline(median[xi], color="b")
            # ax.axhline(median[yi], color="b")
            # ax.plot(median[xi], median[yi], "sb")
            # ax.axvline(mode[xi], color="y")
            # ax.axhline(mode[yi], color="y")
            # ax.plot(mode[xi], mode[yi], "sy")

    loc ="test"+str(i)+"_tnd.png"
    plt.savefig(loc)