from dateutil import parser
import numpy as np
import pandas as pd
from iblutil.numerical import ismember

from brainbox.io.one import SpikeSortingLoader, SessionLoader


def bwm_query(one, alignment_resolved=True, return_details=False, freeze=None):
    """
    Function to query for brainwide map sessions that pass the most important quality controls. Returns a dataframe
    with one row per insertions and columns ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to local or remote database
    alignment_resolved: bool
        Default is True. If True, only returns sessions with resolved alignment, if False returns all sessions with at
        least one alignment
    return_details: bool
        Default is False. If True returns a second output a list containing the full insertion dictionary for all
        insertions returned by the query. Only needed if you need information that is not contained in the bwm_df.
    freeze: {None}
        If None, the database is queried for the current set of pids satisfying the criteria. If a string is specified,
        a fixed set of eids and pids is returned instead of querying the database. The string must be equivalent to the
        name of an official freeze.

    Returns
    -------
    bwm_df: pandas.DataFrame
        BWM sessions to be included in analyses with columns
        ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']
    """

    # if freeze is not None:
    #     try:
    #     except
    #         print(f'{freeze} does not seem to be a dataset freeze.')

    base_query = (
        'session__project__name__icontains,ibl_neuropixel_brainwide_01,'
        'session__json__IS_MOCK,False,'
        'session__qc__lt,50,'
        '~json__qc,CRITICAL,'
        'session__extended_qc__behavior,1,'
        'json__extended_qc__tracing_exists,True,'
    )

    if alignment_resolved:
        base_query += 'json__extended_qc__alignment_resolved,True,'
    else:
        base_query += 'json__extended_qc__alignment_count__gt,0,'

    qc_pass = (
        '~session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
        '~session__extended_qc___task_response_feedback_delays__lt,0.9,'
        '~session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
        '~session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
        '~session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_reward_volumes__lt,0.9,'
        '~session__extended_qc___task_reward_volume_set__lt,0.9,'
        '~session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
        '~session__extended_qc___task_audio_pre_trial__lt,0.9')

    marked_pass = (
        'session__extended_qc___experimenter_task,PASS')

    insertions = list(one.alyx.rest('insertions', 'list', django=base_query + qc_pass))
    insertions.extend(list(one.alyx.rest('insertions', 'list', django=base_query + marked_pass)))

    bwm_df = pd.DataFrame({
        'pid': np.array([i['id'] for i in insertions]),
        'eid': np.array([i['session'] for i in insertions]),
        'probe_name': np.array([i['name'] for i in insertions]),
        'session_number': np.array([i['session_info']['number'] for i in insertions]),
        'date': np.array([parser.parse(i['session_info']['start_time']).date() for i in insertions]),
        'subject': np.array([i['session_info']['subject'] for i in insertions]),
        'lab': np.array([i['session_info']['lab'] for i in insertions]),
    }).sort_values(by=['lab', 'subject', 'date', 'eid'])
    bwm_df.drop_duplicates(inplace=True)
    bwm_df.reset_index(inplace=True, drop=True)

    if return_details:
        return bwm_df, insertions
    else:
        return bwm_df


def load_good_units(one, pid, **kwargs):
    """
    Function to load the cluster information and spike trains for clusters that pass all quality metrics.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to local or remote database
    pid: str
        A probe insertion UUID

    Returns
    -------
    good_clusters: pandas.DataFrame
        Information of clusters for this pid that pass all quality metrics
    good_spikes: dict
        Spike trains associated with good clusters. Dictionary with keys ['depths', 'times', 'clusters', 'amps']
    """
    eid = kwargs.pop('eid', '')
    pname = kwargs.pop('pname', '')
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = spike_loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    iok = clusters_labeled['label'] == 1
    good_clusters = clusters_labeled[iok]

    spike_idx, ib = ismember(spikes['clusters'], good_clusters.index)
    good_clusters.reset_index(drop=True, inplace=True)
    # Filter spike trains for only good clusters
    good_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    good_spikes['clusters'] = good_clusters.index[ib].astype(np.int32)

    return good_spikes, good_clusters


def load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default'):
    """
    Function to load all trials for a given session and create a mask to exclude all trials that have a reaction time
    shorter than min_rt or longer than max_rt or that have NaN for one of the specified events.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to local or remote database
    eid: str
        A session UUID
    min_rt: float
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08.
    max_rt: float
        Maximum admissible reaction time in seconds for a trial to be included. Default is 2.
    nan_exclude: list or 'default'
        List of trial events that cannot be NaN for a trial to be included. If set to 'default' the list is
        ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']

    Returns
    -------
    trials: pandas.DataFrame
        Trials table containing all trials for this session. If complete with columns:
        ['stimOff_times','intervals_bpod_0','intervals_bpod_1','goCueTrigger_times','feedbackType','contrastLeft',
        'contrastRight','rewardVolume','goCue_times','choice','feedback_times','stimOn_times','response_times',
        'firstMovement_times','probabilityLeft', 'intervals_0', 'intervals_1']
    mask: pandas.Series
        Boolean Series to mask trials table for trials that pass specified criteria. True for all trials that should be
        included, False for all trials that should be excluded.
    """

    if nan_exclude == 'default':
        nan_exclude = [
            'stimOn_times',
            'choice',
            'feedback_times',
            'probabilityLeft',
            'firstMovement_times',
            'feedbackType'
        ]

    sess_loader = SessionLoader(one, eid)
    sess_loader.load_trials()

    # Create a mask for trials to exclude
    # Remove trials that are outside the allowed reaction time range
    query = f'(firstMovement_times - stimOn_times < {min_rt}) | (firstMovement_times - stimOn_times > {max_rt})'
    # Remove trials with nan in specified events
    for event in nan_exclude:
        query += f' | {event}.isnull()'

    # Create mask
    mask = ~sess_loader.trials.eval(query)

    return sess_loader.trials, mask
