import numpy as np
import pandas as pd
import functions.utils as dut
import brainbox.io.one as bbone
import models.utils as mut
from pathlib import Path
from functions.utils import save_region_results
from one.api import ONE
from one.api import One
from brainbox.population.decode import get_spike_counts_in_bins
import one.alf.io as alfio
from functions.neurometric import get_neurometric_parameters
from tqdm import tqdm

bin_size_kde = 0.05
out, target_pLefts = dut.get_target_pLeft(nb_trials=100000, nb_sessions=1,
                                          take_out_unbiased=True, bin_size_kde=bin_size_kde)
plt.hist(target_pLefts, bins=np.arange(0, 1, 0.05) + 0.025, density=True)
plt.xlabel('pLeft')
plt.ylabel('empirical density')

import pickle
with open(DECODING_PATH.joinpath('targetpLeft_optBay_%s.pkl' % str(bin_size_kde).replace('.', '_')), 'wb') as f:
    pickle.dump([out[0], out[1]], f)

'''
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold
# target_distribution = pickle.load(f)
# target_pdf = lambda x: target_distribution[0][(x > target_distribution[1]).sum() - 1]

bandwidths = [0.0001, 0.001, 0.01, 0.025, 0.05, 0.1]
nFolds = 10
score = np.zeros([len(bandwidths), nFolds])
kf = KFold(n_splits=nFolds)
for i_b, b in enumerate(bandwidths):
    for ifold, (train_index, test_index) in enumerate(kf.split(x)):
        train, test = x[train_index], x[test_index]
        out = numpy.histogram(train, bins=np.arange(0, 1, b) + b/2, density=True)
        score[i_b, ifold] = np.log(pdf_from_histogram(test, out)).sum()
print(score.mean(axis=1))

bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=KFold(n_splits=20))
grid.fit(x[:, None])
grid.best_params_

factory = openturns.KernelSmoothing()
sample = openturns.Sample(target_pLeft.ravel()[:, None])
bandwidth = factory.computePluginBandwidth(sample)
target_distribution = factory.build(sample, bandwidth)
target_weights = np.array(target_distribution.computePDF(sample)).squeeze()

with open(DECODING_PATH.joinpath('target_distribution_pLeft_optBay.pkl', 'wb')) as f:
    pickle.dump(target_distribution, f)


plt.figure()
plt.hist(target_pLeft.ravel()[:, None], density=True)
plt.plot(sample, target_weights, '+')
'''
