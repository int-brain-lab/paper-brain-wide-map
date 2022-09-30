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


def load_metadata(neural_dtype_path_regex, date=None):
    '''
    Parameters
    ----------
    neural_dtype_path_regex
    date
    Returns
    -------
    the metadata neural_dtype from date if specified, else most recent
    '''
    neural_dtype_paths = glob.glob(neural_dtype_path_regex)
    neural_dtype_dates = [datetime.strptime(p.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S.%f')
                          for p in neural_dtype_paths]
    if date is None:
        path_id = np.argmax(neural_dtype_dates)
    else:
        path_id = np.argmax(np.array(neural_dtype_dates) == date)
    return pickle.load(open(neural_dtype_paths[path_id], 'rb')), neural_dtype_dates[path_id].strftime("%m-%d-%Y_%H:%M:%S")


def compute_mask(trials_df, align_time, time_window, min_len, max_len, no_unbias, min_rt, max_rt, **kwargs):
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
    kwargs

    Returns
    -------
    pd.Series

    """

    # define reaction times
    if 'react_times' not in trials_df.keys():
        trials_df['react_times'] = trials_df.firstMovement_times - trials_df.stimOn_times

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
        tmp = (trials_df[align_time].values[:-1] + time_window[1]) < trials_df.trial_start.values[1:]
        tmp = np.concatenate([tmp, [True]])  # include final trial, no following trials
        mask = mask & tmp

    # get rid of trials where animal does not respond
    mask = mask & (trials_df.choice != 0)

    if kwargs['nb_trials_takeout_end'] > 0:
        mask[-int(kwargs['nb_trials_takeout_end']):] = False

    return mask


def get_save_path(
        pseudo_id, subject, eid, neural_dtype, probe, region, output_path, time_window, today, target,
        add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(neural_dtype, subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    start_tw, end_tw = time_window
    fn = '_'.join([today, region, 'target', target,
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


def check_settings(settings):
    """Error check on pipeline settings.

    Parameters
    ----------
    settings : dict

    Returns
    -------
    dict

    """

    from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
    from braindelphi.decoding.functions.process_targets import optimal_Bayesian

    # options for decoding targets
    target_options_singlebin = [
        'prior',     # some estimate of the block prior
        'choice',    # subject's choice (L/R)
        'feedback',  # correct/incorrect
        'signcont',  # signed contrast of stimulus
    ]
    target_options_multibin = [
        'wheel-vel',
        'wheel-speed',
        'pupil',
        'l-paw-pos',
        'l-paw-vel',
        'l-paw-speed',
        'l-whisker-me',
        'r-paw-pos',
        'r-paw-vel',
        'r-paw-speed',
        'r-whisker-me',
    ]

    # options for behavioral models
    behavior_model_options = {
        'expSmoothing_prevAction': expSmoothing_prevAction,
        'optimal_Bayesian': optimal_Bayesian,
        'oracle': None,
    }

    # options for align events
    align_event_options = [
        'firstMovement_times',
        'goCue_times',
        'stimOn_times',
        'feedback_times',
    ]

    # options for decoder
    decoder_options = {
        'linear': sklm.LinearRegression,
        'lasso': sklm.Lasso,
        'ridge': sklm.Ridge,
        'logistic': sklm.LogisticRegression
    }

    # load default settings
    settings_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'settings_default.yaml')
    params = yaml.safe_load(open(settings_file))

    # update default setting with user-provided settings
    params.update(copy.copy(settings))

    if params['target'] not in target_options_singlebin + target_options_multibin:
        raise NotImplementedError('provided target option \'{}\' invalid; must be in {}'.format(
            params['target'], target_options_singlebin + target_options_multibin
        ))

    if params['model'] not in behavior_model_options.keys():
        raise NotImplementedError('provided beh model option \'{}\' invalid; must be in {}'.format(
            params['model'], behavior_model_options.keys()
        ))

    if params['align_time'] not in align_event_options:
        raise NotImplementedError('provided align event \'{}\' invalid; must be in {}'.format(
            params['align_time'], align_event_options
        ))

    if not params['single_region'] and not params['merge_probes']:
        raise ValueError('full probes analysis can only be done with merged probes')

    if params['compute_neurometric'] and kwargs['target'] != 'signcont':
        raise ValueError('the target should be signcont to compute neurometric curves')

    if len(params['border_quantiles_neurometric']) == 0 and params['model'] != 'oracle':
        raise ValueError(
            'border_quantiles_neurometric must be at least of 1 when behavior model is specified')

    if len(params['border_quantiles_neurometric']) != 0 and params['model'] == 'oracle':
        raise ValueError(
            f'border_quantiles_neurometric must be empty when behavior model is not specified'
            f'- oracle pLeft used'
        )

    # map behavior model string to model class
    if params['model'] == 'logistic' and params['balanced_continuous_target']:
        raise ValueError('you can not have a continuous target with logistic regression')

    params['model'] = behavior_model_options[params['model']]

    # map estimator string to sklm class
    if params['estimator'] == 'logistic':
        params['hyperparam_grid'] = {'C': params['hyperparam_grid']['C']}
    else:
        params['hyperparam_grid'] = {'alpha': params['hyperparam_grid']['alpha']}
    params['estimator'] = decoder_options[params['estimator']]

    params['n_jobs_per_session'] = params['n_pseudo'] // params['n_pseudo_per_job']

    # TODO: settle on 'date' or 'today'
    # update date if not given
    if params['date'] is None or params['date'] == 'today':
        params['date'] = str(date.today())
    params['today'] = params['date']

    # TODO: settle on n_runs or nb_runs
    if 'n_runs' in params:
        params['nb_runs'] = params['n_runs']

    # TODO: settle on align_time or align_event
    params['align_event'] = params['align_time']

    return params


def derivative(y):
    dy = np.zeros(y.shape,np.float)
    dy[0:-1] = np.diff(y)
    dy[-1] = (y[-1] - y[-2])
    return dy
