'''
Test the functions for receptive fields
'''
import numpy as np

# Load RF data (on), select region: VISP
rf_on = np.load('BWM_rf_on.npy')
rf_beryl_label = np.load('BWM_rf_beryl_label.npy')
vis_ind = np.argwhere(rf_beryl_label == 385)[:, 0]
rf_vis = rf_on[vis_ind, :, :]

