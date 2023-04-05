import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from behavior_models.models.utils import format_data as format_data_mut
from behavior_models.models.utils import format_input as format_input_mut
from behavior_models.models.utils import build_path as build_path_mut
from brainwidemap.bwm_loading import load_trials_and_mask


def optimal_Bayesian(act, side):
    '''
    Generates the optimal prior
    Params:
        act (array of shape [nb_sessions, nb_trials]): action performed by the mice of shape
        side (array of shape [nb_sessions, nb_trials]): stimulus side (-1 (right), 1 (left)) observed by the mice
    Output:
        prior (array of shape [nb_sessions, nb_chains, nb_trials]): prior for each chain and session
    '''
    act = torch.from_numpy(act)
    side = torch.from_numpy(side)
    lb, tau, ub, gamma = 20, 60, 100, 0.8
    nb_blocklengths = 100
    nb_typeblocks = 3
    eps = torch.tensor(1e-15)

    alpha = torch.zeros([act.shape[-1], nb_blocklengths, nb_typeblocks])
    alpha[0, 0, 1] = 1
    alpha = alpha.reshape(-1, nb_blocklengths * nb_typeblocks)
    h = torch.zeros([nb_typeblocks * nb_blocklengths])

    # build transition matrix
    b = torch.zeros([nb_blocklengths, nb_typeblocks, nb_typeblocks])
    b[1:][:, 0, 0], b[1:][:, 1, 1], b[1:][:, 2, 2] = 1, 1, 1  # case when l_t > 0
    b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = 1, 1, 1. / 2  # case when l_t = 1
    n = torch.arange(1, nb_blocklengths + 1)
    ref = torch.exp(-n / tau) * (lb <= n) * (ub >= n)
    torch.flip(ref.double(), (0,))
    hazard = torch.cummax(
        ref / torch.flip(torch.cumsum(torch.flip(ref.double(), (0,)), 0) + eps, (0,)), 0)[0]
    tmp = torch.cat(
        (torch.unsqueeze(hazard, -1),
         torch.cat((torch.diag(1 - hazard[:-1]), torch.zeros(nb_blocklengths - 1)[None]), axis=0)),
        axis=-1)  # l_{t-1}, l_t
    transition = eps + torch.transpose(tmp[:, :, None, None] * b[None], 1, 2).reshape(
        nb_typeblocks * nb_blocklengths, -1)

    # likelihood
    lks = torch.hstack([
        gamma * (side[:, None] == -1) + (1 - gamma) * (side[:, None] == 1),
        torch.ones_like(act[:, None]) * 1. / 2,
        gamma * (side[:, None] == 1) + (1 - gamma) * (side[:, None] == -1)
    ])
    to_update = torch.unsqueeze(torch.unsqueeze(act.not_equal(0), -1), -1) * 1

    for i_trial in range(act.shape[-1]):
        # save priors
        if i_trial >= 0:
            if i_trial > 0:
                alpha[i_trial] = torch.sum(torch.unsqueeze(h, -1) * transition, axis=0) \
                    * to_update[i_trial - 1] \
                    + alpha[i_trial - 1] * (1 - to_update[i_trial - 1])
            # else:
            #    alpha = alpha.reshape(-1, nb_blocklengths, nb_typeblocks)
            #    alpha[i_trial, 0, 0] = 0.5
            #    alpha[i_trial, 0, -1] = 0.5
            #    alpha = alpha.reshape(-1, nb_blocklengths * nb_typeblocks)
            h = alpha[i_trial] * lks[i_trial].repeat(nb_blocklengths)
            h = h / torch.unsqueeze(torch.sum(h, axis=-1), -1)
        else:
            if i_trial > 0:
                alpha[i_trial, :] = alpha[i_trial - 1, :]

    predictive = torch.sum(alpha.reshape(-1, nb_blocklengths, nb_typeblocks), 1)
    Pis = predictive[:, 0] * gamma + predictive[:, 1] * 0.5 + predictive[:, 2] * (1 - gamma)

    return 1 - Pis


def compute_target_mask(target, exclude_trials_within_values):
    """
    Computes a boolean array which is False where target indicies should be excluded
    and True otherwise.

    Parameters
    ----------
    target : array of scalar values for each trial.  does not allow for multiple values
             per trial
    exclude_trials_within_values: 2-tuple where each value is either a float of None

    Returns
    ----------
    boolean numpy array

    """
    valwin = exclude_trials_within_values
    target_mask_lb = None if valwin[0] is None else target < valwin[0]
    target_mask_ub = None if valwin[1] is None else target > valwin[1]
    if (not (target_mask_lb is None)) and (not (target_mask_ub is None)):
        target_mask = target_mask_lb | target_mask_ub
    elif not (target_mask_lb is None):
        target_mask = target_mask_lb
    elif not (target_mask_ub is None):
        target_mask = target_mask_ub
    else:
        target_mask = np.ones(len(target), dtype=bool)
    return target_mask


def compute_beh_target(trials_df, metadata, return_raw=False, **kwargs):
    """
    Computes regression target for use with regress_target, using subject, eid, and a string
    identifying the target parameter to output a vector of N_trials length containing the target

    Parameters
    ----------
    target : str
        String in ['prior', 'prederr', 'signcont'], indication model-based prior, prediction error,
        or simple signed contrast per trial
    subject : str
        Subject identity in the IBL database, e.g. KS022
    eids_train : list of str
        list of UUID identifying sessions on which the model is trained.
    eids_test : str
        UUID identifying sessions on which the target signal is computed
    savepath : str
        where the beh model outputs are saved
    behmodel : str
        behmodel to use
    pseudo : bool
        Whether or not to compute a pseudosession result, rather than a real result.
    modeltype : behavior_models model object
        Instantiated object of behavior models. Needs to be instantiated for pseudosession target
        generation in the case of a 'prior' or 'prederr' target.
    beh_data : behavioral data feed to the model when using pseudo-sessions
    return_raw : bool, if True returns tuple where second element is the target without applying
                 a transformation or binarization.  this is useful for computing a mask from target
                 values

    Returns
    -------
    pandas.Series
        Pandas series in which index is trial number, and value is the target
    pandas.Series
        same pandas series , returned only if return_raw is True
    """

    '''
    load/fit a behavioral model to compute target on a single session
    Params:
        eids_train: list of eids on which we train the network
        eid_test: eid on which we want to compute the target signals, only one string
        beh_data_test: if you have to launch the model on beh_data_test.
                       if beh_data_test is explicited, the eid_test will not be considered
        target can be pLeft or signcont. If target=pLeft, it will return the prior predicted by modeltype
                                         if modetype=None, then it will return the actual pLeft (.2, .5, .8)
    '''

    tvec = get_beh_target(trials_df, metadata, remove_old=False, **kwargs)
    tvec_raw = tvec.copy()
    if kwargs['tanh_transform']:
        tvec = np.tanh(5 * tvec) / np.tanh(5)
    if kwargs['binarization_value'] is not None:
        tvec = (tvec > kwargs['binarization_value']) * 1
    if return_raw:
        return tvec, tvec_raw
    return tvec


def get_beh_target(trials_df, metadata, remove_old=False, **kwargs):
    if kwargs['model_parameters'] is None:
        istrained, fullpath = check_bhv_fit_exists(
            metadata['subject'], kwargs['model'], metadata['eids_train'], kwargs['behfit_path'],
            modeldispatcher=kwargs['modeldispatcher'], single_zeta=True)
    else:
        istrained, fullpath = False, ''

    if kwargs['target'] in ['signcont', 'strengthcont']:
        if 'signedContrast' in trials_df.keys():
            out = trials_df['signedContrast'].values
        else:
            out = np.nan_to_num(trials_df.contrastLeft) - np.nan_to_num(trials_df.contrastRight)
        if kwargs['target'] == 'signcont':
            return out
        else:
            return np.abs(out)
    if kwargs['target'] == 'choice':
        return trials_df.choice.values
    if kwargs['target'] == 'feedback':
        return trials_df.feedbackType.values
    elif (kwargs['target'] == 'pLeft') and (kwargs['model'] is None):
        return trials_df.probabilityLeft.values
    elif (kwargs['target'] == 'pLeft') and (kwargs['model'] is optimal_Bayesian):  # bypass fitting and generate priors
        side, stim, act, _ = format_data_mut(trials_df)
        signal = optimal_Bayesian(act, side)
        return signal.numpy().squeeze()

    if ((not istrained) and (kwargs['target'] != 'signcont') and
            (kwargs['model'] is not None) and kwargs['model_parameters'] is None):
        datadict = {'stim_side': [], 'actions': [], 'stimuli': []}
        if 'eids_train' in kwargs.keys() or len(metadata['eids_train']) >= 2:
            raise NotImplementedError('Sorry, this features is not implemented yet')
        for _ in metadata['eids_train']:  # this seems superfluous but this is a relevant structure for when eids_train != [eid]
            side, stim, act, _ = format_data_mut(trials_df)
            datadict['stim_side'].append(side)
            datadict['stimuli'].append(stim)
            datadict['actions'].append(act)
        stimuli, actions, stim_side = format_input_mut(datadict['stimuli'], datadict['actions'], datadict['stim_side'])
        model = kwargs['model'](kwargs['behfit_path'], np.array(metadata['eids_train']), metadata['subject'],
                                actions, stimuli, stim_side, single_zeta=True)
        model.load_or_train(remove_old=remove_old)
    elif (kwargs['target'] != 'signcont') and (kwargs['model'] is not None):
        model = kwargs['model'](kwargs['behfit_path'],
                                metadata['eids_train'],
                                metadata['subject'],
                                actions=None,
                                stimuli=None,
                                stim_side=None, single_zeta=True)
        if kwargs['model_parameters'] is None:
            model.load_or_train(loadpath=str(fullpath))

    # compute signal
    stim_side, stimuli, actions, _ = format_data_mut(trials_df)
    stimuli, actions, stim_side = format_input_mut([stimuli], [actions], [stim_side])
    if kwargs['model_parameters'] is None:
        signal = model.compute_signal(signal='prior' if kwargs['target'] == 'pLeft' else kwargs['target'],
                                      act=actions,
                                      stim=stimuli,
                                      side=stim_side)['prior' if kwargs['target'] == 'pLeft' else kwargs['target']]
    else:
        output = model.evaluate(np.array(list(kwargs['model_parameters'].values()))[None],
                                return_details=True, act=actions, stim=stimuli, side=stim_side)
        signal = output[1]

    tvec = signal.squeeze()
    return tvec


def get_target_data_per_trial_wrapper(
        target_times, target_vals, trials_df, align_event, align_interval, binsize,
        allow_nans=False):
    """Format a single session-wide array of target data into a list of trial-based arrays.

    Note: the bin size of the returned data will only be equal to the input `binsize` if that value
    evenly divides `align_interval`; for example if `align_interval=(0, 0.2)` and `binsize=0.10`,
    then the returned data will have the correct binsize. If `align_interval=(0, 0.2)` and
    `binsize=0.06` then the returned data will not have the correct binsize.

    Parameters
    ----------
    target_times : array-like
        time in seconds for each sample
    target_vals : array-like
        data samples
    trials_df : pd.DataFrame
        requires a column that matches `align_event`
    align_event : str
        event to align interval to
        firstMovement_times | stimOn_times | feedback_times
    align_interval : tuple
        (align_begin, align_end); time in seconds relative to align_event
    binsize : float
        size of individual bins in interval
    allow_nans : bool, optional
        False to skip trials with >0 NaN values in target data

    Returns
    -------
    tuple
        - (list): time in seconds for each trial
        - (list): data for each trial
        - (array-like): mask of good trials (True) and bad trials (False)

    """

    align_times = trials_df[align_event].values
    interval_begs = align_times + align_interval[0]
    interval_ends = align_times + align_interval[1]
    interval_len = align_interval[1] - align_interval[0]

    # split data by trial
    if np.all(np.isnan(interval_begs)) or np.all(np.isnan(interval_ends)):
        print('interval times all nan')
        good_trial = np.nan * np.ones(interval_begs.shape[0])
        target_times_list = []
        target_vals_list = []
        return target_times_list, target_vals_list, good_trial

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    # split data into trials
    idxs_beg = np.searchsorted(target_times, interval_begs, side='right')
    idxs_end = np.searchsorted(target_times, interval_ends, side='left')
    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # interpolate and store
    target_times_list = []
    target_vals_list = []
    good_trial = [None for _ in range(len(target_times_og_list))]
    for i, (target_time, target_vals) in enumerate(zip(target_times_og_list, target_vals_og_list)):

        if len(target_vals) == 0:
            print('target data not present on trial %i; skipping' % i)
            good_trial[i] = False
            target_times_list.append(None)
            target_vals_list.append(None)
            continue
        if np.sum(np.isnan(target_vals)) > 0 and not allow_nans:
            print('nans in target data on trial %i; skipping' % i)
            good_trial[i] = False
            target_times_list.append(None)
            target_vals_list.append(None)
            continue
        if np.isnan(interval_begs[i]) or np.isnan(interval_ends[i]):
            print('bad trial interval data on trial %i; skipping' % i)
            good_trial[i] = False
            target_times_list.append(None)
            target_vals_list.append(None)
            continue
        if np.abs(interval_begs[i] - target_time[0]) > binsize:
            print('target data starts too late on trial %i; skipping' % i)
            good_trial[i] = False
            target_times_list.append(None)
            target_vals_list.append(None)
            continue
        if np.abs(interval_ends[i] - target_time[-1]) > binsize:
            print('target data ends too early on trial %i; skipping' % i)
            good_trial[i] = False
            target_times_list.append(None)
            target_vals_list.append(None)
            continue

        # resample signal in desired bins
        # using `interval_begs[i] + binsize` forces the interpolation to sample the continuous
        # signal at the *end* (or right side) of each bin; this way the spikes in a given bin will
        # fully precede the corresponding target sample for that same bin.
        x_interp = np.linspace(interval_begs[i] + binsize, interval_ends[i], n_bins)
        if len(target_vals.shape) > 1 and target_vals.shape[1] > 1:
            n_dims = target_vals.shape[1]
            y_interp_tmps = []
            for n in range(n_dims):
                y_interp_tmps.append(interp1d(
                    target_time, target_vals[:, n], kind='linear',
                    fill_value='extrapolate')(x_interp))
            y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
        else:
            y_interp = interp1d(
                target_time, target_vals, kind='linear', fill_value='extrapolate')(x_interp)

        target_times_list.append(x_interp)
        target_vals_list.append(y_interp)
        good_trial[i] = True

    return target_times_list, target_vals_list, np.array(good_trial)


def load_behavior(target, sess_loader):
    """

    Parameters
    ----------
    target : str
        'wheel-vel' | 'wheel-speed' | 'l-whisker-me' | 'r-whisker-me'
    sess_loader : from brainbox.io.one.SessionLoader

    Returns
    -------
    dict
        'times': timestamps for behavior signal
        'values': associated values
        'skip': bool, True if there was an error loading data

    """
    try:
        if target == 'wheel-vel':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': sess_loader.wheel['velocity'].to_numpy()
            }
        elif target == 'wheel-speed':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': np.abs(sess_loader.wheel['velocity'].to_numpy())
            }
        elif target == 'l-whisker-me':
            sess_loader.load_motion_energy(views=['left'])
            beh_dict = {
                'times': sess_loader.motion_energy['leftCamera']['times'].to_numpy(),
                'values': sess_loader.motion_energy['leftCamera']['whiskerMotionEnergy'].to_numpy()
            }
        elif target == 'r-whisker-me':
            sess_loader.load_motion_energy(views=['right'])
            beh_dict = {
                'times': sess_loader.motion_energy['rightCamera']['times'].to_numpy(),
                'values': sess_loader.motion_energy['rightCamera']['whiskerMotionEnergy'].to_numpy()
            }
        else:
            raise NotImplementedError

        beh_dict['skip'] = False

    except BaseException as e:
        print('error loading %s data' % target)
        print(e)
        beh_dict = {'times': None, 'values': None, 'skip': True}

    return beh_dict


def get_target_variable_in_df(
        one, eid, sess_loader, target, align_time, time_window, binsize, min_rt, max_rt, min_len,
        max_len, exclude_unbiased_trials):
    """Return a trials dataframe with additional behavioral data as one array per trial.

    Note that the returned trials are masked:
    * trial masking: trials are removed if reaction times are too short/long, the animal did not
      make a choice, etc. See brainwidemap.bwm_loading import load_trials_and_mask for more
      documentation
    * target masking: trials are removed if the target data contains nans, etc. See
      brainwidemap.decoding.functions.process_targets.get_target_data_per_trial_wrapper for more
      documentation

    Parameters
    ----------
    one : ONE object
    eid : str
    sess_loader : brainbox.io.one.SessionLoader object
    target : str
    align_time : str
    time_window : array-like
    binsize : float
    min_rt : float
    max_rt : float
    min_len : float
    max_len : float
    exclude_unbiased_trials : bool

    Returns
    -------
    pandas.DataFrame

    """

    # load trial data
    trials_df, trials_mask = load_trials_and_mask(
        one=one, eid=eid, sess_loader=sess_loader,
        min_rt=min_rt, max_rt=max_rt, min_trial_len=min_len, max_trial_len=max_len,
        exclude_nochoice=True, exclude_unbiased=exclude_unbiased_trials)

    # load behavior data
    beh_dict = load_behavior(target=target, sess_loader=sess_loader)
    if beh_dict['skip']:
        raise Exception("Error loading %s data" % target)

    # split behavior data into trials
    _, target_vals_list, target_mask = get_target_data_per_trial_wrapper(
        beh_dict['times'], beh_dict['values'], trials_df, align_time, time_window, binsize)

    if len(target_vals_list) == 0:
        return None
    else:
        trials_df[target] = target_vals_list
        # return only "good" trials
        mask = trials_mask & target_mask
        return trials_df[mask]


def check_bhv_fit_exists(subject, model, eids, resultpath, modeldispatcher, single_zeta):
    '''
    subject: subject_name
    eids: sessions on which the model was fitted
    check if the behavioral fits exists
    return Bool and filename
    '''
    if model not in modeldispatcher.keys():
        raise KeyError('Model is not an instance of a model from behavior_models')
    path_results_mouse = 'model_%s_' % modeldispatcher[model] + 'single_zeta_' * single_zeta
    trunc_eids = [eid.split('-')[0] for eid in eids]
    filen = build_path_mut(path_results_mouse, trunc_eids)
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    fullpath = subjmodpath.joinpath(filen)
    return os.path.exists(fullpath), fullpath

def transform_data_for_decoding(ys, Xs):
    # transform target data into standard format: list of np.ndarrays
    if isinstance(ys, np.ndarray):
        # input is single numpy array
        ys = [np.array([y]) for y in ys]
    elif isinstance(ys, list) and ys[0].shape == ():
        # input is list of float instead of list of np.ndarrays
        ys = [np.array([y]) for y in ys]
    elif isinstance(ys, pd.Series):
        # input is a pandas Series
        ys = ys.to_numpy()
        ys = [np.array([y]) for y in ys]

    # transform neural data into standard format: list of np.ndarrays
    if isinstance(Xs, np.ndarray):
        Xs = [x[None, :] for x in Xs]

    return ys, Xs

def logisticreg_criteria(ys, MIN_UNIQUE_COUNTS=3):
    #if estimator == sklm.LogisticRegression:
    # target must have 2 classes and at least 3 of each class to allow
    #    for at least 2 classes in each of the train and validate sub-folds
    ys = transform_data_for_decoding(ys, [])[0]
    y_uniquecounts = np.unique(ys, return_counts=True)[1]
    #print('failed outer fold, target unique counts:', y_uniquecounts)
    return len(y_uniquecounts)==2 and np.min(y_uniquecounts)>=MIN_UNIQUE_COUNTS
