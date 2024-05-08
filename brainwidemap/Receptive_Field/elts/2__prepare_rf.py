'''
Prepare RF for testing
'''
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Load RFs pre-computed
# from https://drive.google.com/drive/folders/17QhM46ZS0PiHFwyguxHl5srR-O05cYJL
DATA_PATH = Path('/Users/gaelle/Downloads')
SAVE_PATH = Path('/Users/gaelle/Documents/Git/int-brain-lab/paper-brain-wide-map/brainwidemap/'
                 'test/fixtures')
# TODO SAVE_PATH: Path(__file__).parent not working
##
# Load RF data (on), select region: VISP
rf_on = np.load(DATA_PATH.joinpath('BWM_rf_on.npy'))
rf_beryl_label = np.load(DATA_PATH.joinpath('BWM_rf_beryl_label.npy'))
rf_pid = pd.read_csv(DATA_PATH.joinpath('BWM_rf_pid.csv'))
rf_cluid = np.load(DATA_PATH.joinpath('BWM_rf_QC_cluster_id.npy'))

# Find units in visual area
vis_ind = np.argwhere(rf_beryl_label == 385)[:, 0]
# Keep only those indices
rf_on = rf_on[vis_ind, :, :]
rf_pid = rf_pid.loc[vis_ind].reset_index().drop(columns=['index'])
rf_cluid = rf_cluid[vis_ind]

##
# Select and plot units that will be used as test
units = [106, 108]
titles = ['significant', 'not significant']
fig, axs = plt.subplots(1, len(units))

for i_unit, unit in enumerate(units):
    # Compute z-score of RF:
    rf_z = rf_on[unit, :, :] - np.mean(rf_on[unit, :, :])
    # plot RF
    plt.axes(axs[i_unit])
    plt.imshow(rf_z)
    plt.colorbar()
    plt.title(titles[i_unit])

##
# Make smaller matrix / metadata containing RF / information about only these units
rf_matrix = rf_on[units, :, :]

rf_meta = rf_pid.copy()
rf_meta = rf_meta.loc[units]
rf_meta['cluster_id'] = rf_cluid[units]

# Save
np.save(SAVE_PATH.joinpath('rf_matrix.npy'), rf_matrix)
rf_meta.to_csv(SAVE_PATH.joinpath('rf_meta.csv'))
