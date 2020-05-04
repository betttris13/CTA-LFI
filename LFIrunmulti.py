import numpy as np
import math
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
from multiprocessing import Pool
import time
from mpi4py import MPI
import sys
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

# Run recostruction over selected range.
def reconstruct(event):
    print("Starting event:", event)

    # Get event form data set
    compressed_data = np.array(test[features])[event]
    params = np.array(test[theta])[event:event+1]
    
    # Define posterior and prior for use by sampler. Prior should match LFI prior.
    def posterior(x):
        return DelfiEnsemble.log_posterior_stacked(x, compressed_data)[0][0]

    def pr(u):
        return pRange*u+lower
    
    # Run sampler and extract samples and statistical values.
    sampler = dynesty.NestedSampler(posterior, pr, 5)
    sampler.run_nested(print_progress=False)
    posterior_samples = sampler.results.samples

    mean = np.mean(posterior_samples, axis = 0)
    median = np.median(posterior_samples, axis = 0)
    v = np.argmax(sampler.results.logl) # Unssure if this is working. Almost certian there is a better way.
    mode = sampler.results.samples[v]
    name = str(event)+"posterior.npy"
    np.save(name, posterior_samples)
    print("Done event:", event)

stime = time.time()

# Index of first and last (inclusive) image to be run. I.e. run over range [first,last].
first = 0
last = 0

start = 0
stop = 0

if len(sys.argv) == 5:
    first = int(sys.argv[1])
    last = int(sys.argv[2])
    rank = int(sys.argv[4])
    size = int(sys.argv[3])

elif len(sys.argv) == 3:
    first = int(sys.argv[1])
    last = int(sys.argv[2])
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

else:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

r = last-first+1
batchsize = r/size
batchsize = math.ceil(batchsize)
start = first + rank*batchsize
stop = start + batchsize
if stop > (last+1):
    stop = last+1

print("Initilizing ", rank, " with batchsize ", batchsize, " from event ", start, " to ", (stop-1))

for i in range(start,stop):
    reconstruct(i)

ftime = time.time()
print("Runtime: ", (ftime-stime))
print("Estimated TTC per event:", str((ftime-stime)/(r)))