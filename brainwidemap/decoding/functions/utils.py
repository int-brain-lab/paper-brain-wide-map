import copy
from datetime import date
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import sklearn.linear_model as sklm
import glob
from datetime import datetime
import yaml


def compute_mask(trials_df, align_time, time_window, min_len, max_len, no_unbias, min_rt, max_rt):
    """Create a mask that denotes "good" trials which will be used for further analysis.

    Parameters
    ----------
    trials_df : dict
        contains relevant trial information like goCue_times, firstMovement_times, etc.
    align_time : str
        event in trial on which to align intervals
        'firstMovement_times' | 'stimOn_times' | 'feedback_times'
    time_window : tuple
        (window_start, window_end), relative to align_time
    min_len : float, optional
        minimum length of trials to keep (seconds), bypassed if trial_start column not in trials_df
    max_len : float, original
        maximum length of trials to keep (seconds), bypassed if trial_start column not in trials_df
    no_unbias : bool
        True to remove unbiased block trials, False to keep them
    min_rt : float
        minimum reaction time; trials with fast reactions will be removed

    Returns
    -------
    pd.Series

    """

    # define reaction times
    if 'react_times' not in trials_df.keys():
        trials_df.loc[:, 'react_times'] = trials_df.firstMovement_times - trials_df.stimOn_times

    # successively build a mask that defines which trials we want to keep

    # ensure align event is not a nan
    mask = trials_df[align_time].notna()

    # ensure animal has moved
    mask = mask & trials_df.firstMovement_times.notna()

    # get rid of unbiased trials
    if no_unbias:
        mask = mask & (trials_df.probabilityLeft != 0.5).values

    # keep trials with reasonable reaction times
    if min_rt is not None:
        mask = mask & (~(trials_df.react_times < min_rt)).values
    if max_rt is not None:
        mask = mask & (~(trials_df.react_times > max_rt)).values

    if 'goCue_times' in trials_df.columns and max_len is not None and min_len is not None:
        # get rid of trials that are too short or too long
        start_diffs = trials_df.goCue_times.diff()
        start_diffs.iloc[0] = 2
        mask = mask & ((start_diffs > min_len).values & (start_diffs < max_len).values)

        # get rid of trials with decoding windows that overlap following trial
        tmp = (trials_df[align_time].values[:-1] + time_window[1]) < trials_df['intervals_0'].values[1:]
        tmp = np.concatenate([tmp, [True]])  # include final trial, no following trials
        mask = mask & tmp

    # get rid of trials where animal does not respond
    mask = mask & (trials_df.choice != 0)

    return mask


def get_save_path(
        pseudo_id, subject, eid, neural_dtype, probe, region, output_path, time_window, date,
        target, add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(neural_dtype, subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    start_tw, end_tw = time_window
    fn = '_'.join([date, region, 'target', target,
                   'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_'),
                   'pseudo_id', str(pseudo_id), add_to_saving_path]) + '.pkl'
    save_path = probefolder.joinpath(fn)
    return save_path


def save_region_results(fit_result, pseudo_id, subject, eid, probe, region, n_units, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    outdict = {
        'fit': fit_result, 'pseudo_id': pseudo_id, 'subject': subject, 'eid': eid, 'probe': probe,
        'region': region, 'N_units': n_units
    }
    fw = open(save_path, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return save_path
