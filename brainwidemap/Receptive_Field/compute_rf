import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import leastsq

import pandas as pd



import numpy as np
#from oneibl.one import ONE
from one.api import ONE
import brainbox.io.one as bbone

from math import *
import sys
import scipy.stats as scist
from os import path

import pandas as pd

from scipy.stats import rankdata



######



one = ONE()
one.refresh_cache('refresh')

from brainbox.population.decode import get_spike_counts_in_bins

from brainbox.io.one import SpikeSortingLoader

from ibllib.atlas import AllenAtlas

ba = AllenAtlas()


from brainbox.task.closed_loop import generate_pseudo_blocks




from iblutil.numerical import ismember



from brainwidemap import bwm_query

from pathlib import Path
from brainbox.io.one import SessionLoader

from brainwidemap import bwm_units, bwm_query, load_good_units, load_trials_and_mask, filter_sessions, \
    download_aggregate_tables

# Specify a path to download the cluster and trials tables
local_path = Path.home().joinpath('bwm_examples')
local_path.mkdir(exist_ok=True)



def load_passive_rfmap(eid, one=None):
    """
    For a given eid load in the passive receptive field mapping protocol data

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : oneibl.one.OneAlyx, optional
        An instance of ONE (may be in 'local' - offline - mode)

    Returns
    -------
    one.alf.io.AlfBunch
        Passive receptive field mapping data
    """
    one = one or ONE()

    # Load in the receptive field mapping data
    rf_map = one.load_object(eid, obj='passiveRFM', collection='alf')
    frames = np.fromfile(one.load_dataset(eid, '_iblrig_RFMapStim.raw.bin',
                                          collection='raw_passive_data'), dtype="uint8")
    y_pix, x_pix = 15, 15
    frames = np.transpose(np.reshape(frames, [y_pix, x_pix, -1], order="F"), [2, 1, 0])
    rf_map['frames'] = frames

    return rf_map

def get_on_off_times_and_positions(rf_map):
    """

    Prepares passive receptive field mapping into format for analysis
    Parameters
    ----------
    rf_map: output from brainbox.io.one.load_passive_rfmap

    Returns
    -------
    rf_map_times: time of each receptive field map frame np.array(len(stim_frames)
    rf_map_pos: unique position of each pixel on screen np.array(len(x_pos), len(y_pos))
    rf_stim_frames: for each pixel on screen stores array of stimulus frames where stim onset
    occurred. For both white squares 'on' and black squares 'off'

    """

    rf_map_times = rf_map['times']
    rf_map_frames = rf_map['frames'].astype('float')

    gray = np.median(rf_map_frames)

    x_bin = rf_map_frames.shape[1]
    y_bin = rf_map_frames.shape[2]

    stim_on_frames = np.zeros((x_bin * y_bin, 1), dtype=np.ndarray)
    stim_off_frames = np.zeros((x_bin * y_bin, 1), dtype=np.ndarray)
    rf_map_pos = np.zeros((x_bin * y_bin, 2), dtype=int)

    i = 0
    for x_pos in np.arange(x_bin):
        for y_pos in np.arange(y_bin):

            pixel_val = rf_map_frames[:, x_pos, y_pos] - gray
            pixel_non_grey = np.where(pixel_val != 0)[0]
            # Find cases where the frame before was gray (i.e when the stim came on)
            frame_change = np.where(rf_map_frames[pixel_non_grey - 1, x_pos, y_pos] == gray)[0]

            stim_pos = pixel_non_grey[frame_change]

            # On stimulus, white squares
            on_pix = np.where(pixel_val[stim_pos] > 0)[0]
            stim_on = stim_pos[on_pix]
            stim_on_frames[i, 0] = stim_on

            off_pix = np.where(pixel_val[stim_pos] < 0)[0]
            stim_off = stim_pos[off_pix]
            stim_off_frames[i, 0] = stim_off

            rf_map_pos[i, :] = [x_pos, y_pos]
            i += 1

    rf_stim_frames = {}
    rf_stim_frames['on'] = stim_on_frames
    rf_stim_frames['off'] = stim_off_frames

    return rf_map_times, rf_map_pos, rf_stim_frames

from iblutil.numerical import bincount2D

def get_rf_map_cluster(rf_map_times, rf_map_pos, rf_stim_frames, spike_times, spike_clusters,
                          t_bin=0.01, d_bin=0, pre_stim=0.0, post_stim=0.15, y_lim=None,
                          x_lim=None):
    """
    Compute receptive field map for each stimulus onset binned across depth
    Parameters
    ----------
    rf_map_times
    rf_map_pos
    rf_stim_frames
    spike_times: array of spike times
    spike_depths: array of spike depths along probe
    t_bin: bin size along time dimension
    pre_stim: time period before rf map stim onset to epoch around
    post_stim: time period after rf map onset to epoch around
    y_lim: values to limit to in depth direction
    x_lim: values to limit in time direction

    Returns
    -------
    rfmap: receptive field map for 'on' 'off' stimuli.
    Each rfmap has shape (depths, x_pos, y_pos, epoch_window)
    depths: cluster id which receptive field map has been computed
    """

    binned_array, times, depths = bincount2D(spike_times, spike_clusters, t_bin, d_bin,
                                             ylim=y_lim, xlim=x_lim)

    x_bin = len(np.unique(rf_map_pos[:, 0]))
    y_bin = len(np.unique(rf_map_pos[:, 1]))
    n_bins = int((pre_stim + post_stim) / t_bin)

    rf_map = {}

    for stim_type, stims in rf_stim_frames.items():
        _rf_map = np.zeros(shape=(depths.shape[0], x_bin, y_bin, n_bins))
        for pos, stim_frame in zip(rf_map_pos, stims):

            x_pos = pos[0]
            y_pos = pos[1]

            # Case where there is no stimulus at this position
            if len(stim_frame[0]) == 0:
                _rf_map[:, x_pos, y_pos, :] = np.zeros((depths.shape[0], n_bins))
                continue

            stim_on_times = rf_map_times[stim_frame[0]]
            stim_intervals = np.c_[stim_on_times - pre_stim, stim_on_times + post_stim]

            out_intervals = stim_intervals[:, 1] > times[-1]
            idx_intervals = np.searchsorted(times, stim_intervals)[np.invert(out_intervals)]

            # Case when no spikes during the passive period
            if idx_intervals.shape[0] == 0:
                avg_stim_trials = np.zeros((depths.shape[0], n_bins))
            else:
                stim_trials = np.zeros((depths.shape[0], n_bins, idx_intervals.shape[0]))
                for i, on in enumerate(idx_intervals):
                    stim_trials[:, :, i] = binned_array[:, on[0]:on[1]]
                avg_stim_trials = np.mean(stim_trials, axis=2)

            _rf_map[:, x_pos, y_pos, :] = avg_stim_trials/t_bin
            #_rf_map[:, x_pos, y_pos, :] = avg_stim_trials

        rf_map[stim_type] = _rf_map
        

    return rf_map, depths


def compute_rf_session(pid, eid):
# compute average rf for each session, returen rf_on/off[number of units, x-position, y-position]   
    

    rf_map=load_passive_rfmap(eid)
    rf_map_times, rf_map_pos, rf_stim_frames=get_on_off_times_and_positions(rf_map)

    


    #spikes, clusters = load_good_units(one, pid, compute_metrics=True)
    
    #temp_spikes, temp_clusters = load_good_units(one, pid, compute_metrics=False)
    #clusters = temp_clusters[temp_clusters['uuids'].isin(list(units_df.uuids))]
    #spike_idx, ib = ismember(temp_spikes['clusters'], clusters.index)
    #clusters.reset_index(drop=True, inplace=True)

    #spikes = {k: v[spike_idx] for k, v in temp_spikes.items()}
    #spikes['clusters'] = clusters.index[ib].astype(np.int32)
    
    
    spikes, clusters = load_good_units(one, pid, compute_metrics=False)

    
    spike_times=spikes['times']
    spike_clusters=spikes['clusters']
    rf_map, rf_clusters = get_rf_map_cluster(rf_map_times, rf_map_pos, rf_stim_frames, spike_times, spike_clusters,t_bin=0.01, d_bin=0, pre_stim=0.0, post_stim=0.15, y_lim=None,x_lim=None)
    
    
    ave_rf_map_on=np.nanmean(rf_map['on'],axis=3)
    ave_rf_map_off=np.nanmean(rf_map['off'],axis=3)
    
    
    area_label=clusters['atlas_id'][rf_clusters].to_numpy()
    
    beryl_label=ba.regions.remap(area_label, source_map='Allen', target_map='Beryl')
    ############ return cluster id ########################
    QC_cluster_id=clusters['cluster_id'][rf_clusters].to_numpy()    
    
    
            
                  
    return   ave_rf_map_on, ave_rf_map_off, area_label, beryl_label, QC_cluster_id
  
