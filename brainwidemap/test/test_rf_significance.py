'''
Test the functions for receptive fields
'''
import numpy as np
from pathlib import Path

# Load from https://drive.google.com/drive/folders/17QhM46ZS0PiHFwyguxHl5srR-O05cYJL
# TEST_PATH = Path(__file__).parent.joinpath('fixtures')
TEST_PATH = Path('Users/Gaelle/Downloads/')

TEST_DATA = TEST_PATH.joinpath('BWM_rf_on.npy')

# Load RF data (on), select region: VISP
rf_on = np.load('BWM_rf_on.npy')
rf_beryl_label = np.load('BWM_rf_beryl_label.npy')
vis_ind = np.argwhere(rf_beryl_label == 385)[:, 0]
rf_vis = rf_on[vis_ind, :, :]

