'''
Test the functions for receptive fields
'''
import numpy as np
from pathlib import Path
import pandas as pd
from receptive_field.rf_significance import fit_2d_gaussian_stats

__file__ = Path('/Users/gaelle/Documents/Git/int-brain-lab/paper-brain-wide-map/brainwidemap/'
                'test/test_rf_significance.py')
# TODO delete the above - should not be needed, but __file__ does not work ATM

TEST_PATH = Path(__file__).parent.joinpath('fixtures')

# Load RF data
rf_matrix = np.load(TEST_PATH.joinpath('rf_matrix.npy'))
rf_meta = pd.read_csv(TEST_PATH.joinpath('rf_meta.csv'))

# Tests values
p_value_test = [0.001,
                0.865]
rsq_test = [0.5625712576099198,
            0.01381862262034228]
fit_params_test = [np.array([24.06780504,  4.83560032, 10.41318107, -0.88336625,  1.38241058]),
                   np.array([2.40647432,  9.0003651, 13.99999986,  0.10337,  0.06264482])]
##
n_unit = rf_meta.shape[0]
for i_unit in range(0, n_unit):
    # Compute p-value, Rsq value, and fitting parameters of 2D guassian
    p_value, rsq, fit_params = fit_2d_gaussian_stats(rf_matrix, nShuffle=1000)
    np.testing.assert_equal(p_value, p_value_test[i_unit])
    np.testing.assert_equal(rsq, rsq_test[i_unit])
    np.testing.assert_equal(fit_params_test, fit_params_test[i_unit])
