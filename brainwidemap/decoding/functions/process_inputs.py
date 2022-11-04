import numpy as np
import scipy

from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.processing import bincount2D


def preprocess_ephys(reg_clu_ids, neural_df, trials_df, **kwargs):
    """Format a single session-wide array of spikes into a list of trial-based arrays.

    Parameters
    ----------
    reg_clu_ids : array-like
        array of cluster ids for each spike
    neural_df : pd.DataFrame
        keys: 'spk_times', 'spk_clu', 'clu_regions', 'clu_qc', 'clu_df'
    trials_df : pd.DataFrame
        columns: 'choice', 'feedback', 'pLeft', 'firstMovement_times', 'stimOn_times',
        'feedback_times'
    kwargs
        align_time : str
            event in trial on which to align intervals
            'firstMovement_times' | 'stimOn_times' | 'feedback_times'
        time_window : tuple
            (window_start, window_end), relative to align_time
        binsize : float, optional
            size of bins in seconds for multi-bin decoding
        n_bins_lag : int, optional
            number of lagged bins to use for predictors for multi-bin decoding

    Returns
    -------
    list
        each element is a 2D numpy.ndarray for a single trial of shape (n_bins, n_clusters)

    """

    # compute time intervals for each trial
    intervals = np.vstack([
        trials_df[kwargs['align_time']] + kwargs['time_window'][0],
        trials_df[kwargs['align_time']] + kwargs['time_window'][1]
    ]).T
    
    # subselect spikes for this region
    spikemask = np.isin(neural_df['spk_clu'], reg_clu_ids)
    regspikes = neural_df['spk_times'][spikemask]
    regclu = neural_df['spk_clu'][spikemask]

    # for each trial, put spiking data into a 2D array; collect trials in a list
    trial_len = kwargs['time_window'][1] - kwargs['time_window'][0]
    binsize = kwargs.get('binsize', trial_len)
    # TODO: can likely combine get_spike_counts_in_bins and get_spike_data_per_trial
    # added second condition in if statement if n_bins_lag is None, gives error otherwise (bensonb)
    if trial_len / binsize == 1.0 or (kwargs['n_bins_lag'] is None):
        # one vector of neural activity per trial
        binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
        binned = binned.T  # binned is a 2D array
        binned_list = [x[None, :] for x in binned]
    else:
        # multiple vectors of neural activity per trial
        # moved interval_len definintion into this condition so that when n_bins_lag is None it doesn't cause error
        interval_len = (
            kwargs['time_window'][1] - kwargs['time_window'][0]
            + kwargs['n_bins_lag'] * kwargs['binsize'])
        spike_times_list, binned_array = get_spike_data_per_trial(
            regspikes, regclu,
            interval_begs=intervals[:, 0] - kwargs['n_bins_lag'] * kwargs['binsize'],
            interval_ends=intervals[:, 1],
            interval_len=interval_len,
            binsize=kwargs['binsize'])
        binned_list = [x.T for x in binned_array]

    return binned_list


def get_spike_data_per_trial(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    """Select spiking data for specified interval on each trial.

    Parameters
    ----------
    times : array-like
        time in seconds for each spike
    clusters : array-like
        cluster id for each spike
    interval_begs : array-like
        beginning of each interval in seconds
    interval_ends : array-like
        end of each interval in seconds
    interval_len : float
    binsize : float
        width of each bin in seconds

    Returns
    -------
    tuple
        - (list): time in seconds for each trial
        - (list): data for each trial of shape (n_clusters, n_bins)

    """

    n_trials = len(interval_begs)

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    binned_spikes = np.zeros((n_trials, n_clusters_in_region, n_bins))
    spike_times_list = []
    for tr, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
        # just get spikes for this region/trial
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            if np.isnan(t_beg) or np.isnan(t_end):
                t_idxs = np.nan * np.ones(n_bins)
            else:
                t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)

        # update data block
        binned_spikes[tr, idxs_tmp, :] += binned_spikes_tmp[:, :n_bins]
        spike_times_list.append(t_idxs[:n_bins])

    return spike_times_list, binned_spikes


def build_predictor_matrix(array, n_lags, return_valid=True):
    """Build predictor matrix with time-lagged datapoints.

    Parameters
    ----------
    array : np.ndarray
        shape (n_time, n_clusters)
    n_lags : int
        number of lagged timepoints (includes zero lag)
    return_valid : bool, optional
        True to crop first n_lags rows, False to leave all

    Returns
    -------
    np.ndarray
        shape (n_time - n_lags, n_clusters * (n_lags + 1)) if return_valid==True
        shape (n_time, n_clusters * (n_lags + 1)) if return_valid==False

    """
    if n_lags < 0:
        raise ValueError('`n_lags` must be >=0, not {}'.format(n_lags))
    mat = np.hstack([np.roll(array, i, axis=0) for i in range(n_lags + 1)])
    if return_valid:
        mat = mat[n_lags:]
    return mat


def select_ephys_regions(regressors, beryl_reg, region, **kwargs):
    """Select units based on QC criteria and brain region."""
    qc_pass = (regressors['clu_qc']['label'] >= kwargs['qc_criteria'])
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
    return reg_clu_ids
