from dateutil import parser
import json
import numpy as np
import pandas as pd
from pathlib import Path

from brainbox.behavior import training
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.atlas import BrainRegions
from iblutil.numerical import ismember
from one.alf import spec
from one.remote import aws

import brainwidemap


def bwm_query(one=None, alignment_resolved=True, return_details=False, freeze='2023_12_bwm_release'):
    """
    Function to query for brainwide map sessions that pass the most important quality controls. Returns a dataframe
    with one row per insertions and columns ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database. Only required if freeze=None.
    alignment_resolved: bool
        Default is True. If True, only returns sessions with resolved alignment, if False returns all sessions with at
        least one alignment
    return_details: bool
        Default is False. If True returns a second output a list containing the full insertion dictionary for all
        insertions returned by the query. Only needed if you need information that is not contained in the bwm_df.
    freeze: {None, 2022_10_initial, 2022_10_update, 2022_10_bwm_release, 2023_12_bwm_release}
        Default is 2023_12_bwm_release. If None, the database is queried for the current set of pids satisfying the
        criteria. If a string is specified, a fixed set of eids and pids is returned instead of querying the database.

    Returns
    -------
    bwm_df: pandas.DataFrame
        BWM sessions to be included in analyses with columns
        ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']
    insertions: list
        Only returned if return_details=True. List of dictionaries with details for each insertions.
    """

    # If a freeze is requested just try to load the respective file
    if freeze is not None:
        if return_details is True:
            print('Cannot return details when using a data freeze. Returning only main dataframe.')

        fixtures_path = Path(brainwidemap.__file__).parent.joinpath('fixtures')
        freeze_file = fixtures_path.joinpath(f'{freeze}.csv')
        assert freeze_file.exists(), f'{freeze} does not seem to be a valid freeze.'

        print(f'Loading bwm_query results from fixtures/{freeze}.csv')
        bwm_df = pd.read_csv(freeze_file, index_col=0)
        bwm_df['date'] = [parser.parse(i).date() for i in bwm_df['date']]

        return bwm_df

    # Else, query the database
    assert one is not None, 'If freeze=None, you need to pass an instance of one.api.ONE'
    base_query = (
        'session__projects__name__icontains,ibl_neuropixel_brainwide_01,'
        '~session__json__IS_MOCK,True,'
        'session__qc__lt,50,'
        'session__extended_qc__behavior,1,'
        '~json__qc,CRITICAL,'
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


def load_good_units(one, pid, compute_metrics=False, qc=1., **kwargs):
    """
    Function to load the cluster information and spike trains for clusters that pass all quality metrics.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    pid: str
        A probe insertion UUID
    compute_metrics: bool
        If True, force SpikeSortingLoader.merge_clusters to recompute the cluster metrics. Default is False
    qc: float
        Quality threshold to be used to select good clusters. Default is 1.0
    kwargs:
        Keyword arguments passed to SpikeSortingLoader upon initiation. Specifically, if one instance offline,
        you need to pass 'eid' and 'pname' here as they cannot be inferred from pid in offline mode.

    Returns
    -------
    good_spikes: dict
        Spike trains associated with good clusters. Dictionary with keys ['depths', 'times', 'clusters', 'amps']
    good_clusters: pandas.DataFrame
        Information of clusters for this pid that pass all quality metrics
    """
    eid = kwargs.pop('eid', '')
    pname = kwargs.pop('pname', '')
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = spike_loader.load_spike_sorting(revision="2024-05-06", good_units=True)
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics).to_df()
    iok = clusters_labeled['label'] >= qc
    good_clusters = clusters_labeled[iok]

    spike_idx, ib = ismember(spikes['clusters'], good_clusters.index)
    good_clusters.reset_index(drop=True, inplace=True)
    # Filter spike trains for only good clusters
    good_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    good_spikes['clusters'] = good_clusters.index[ib].astype(np.int32)

    return good_spikes, good_clusters


def merge_probes(spikes_list, clusters_list):
    """
    Merge spikes and clusters information from several probes as if they were recorded from the same probe.
    This can be used to account for the fact that data from the probes recorded in the same session are not
    statistically independent as they have the same underlying behaviour.

    NOTE: The clusters dataframe will be re-indexed to avoid duplicated indices. Accordingly, spikes['clusters']
    will be updated. To unambiguously identify clusters use the column 'uuids'

    Parameters
    ----------
    spikes_list: list of dicts
        List of spike dictionaries as loaded by SpikeSortingLoader or brainwidemap.load_good_units
    clusters_list: list of pandas.DataFrames
        List of cluster dataframes as loaded by SpikeSortingLoader.merge_clusters or brainwidemap.load_good_units

    Returns
    -------
    merged_spikes: dict
        Merged and time-sorted spikes in single dictionary, where 'clusters' is adjusted to index into merged_clusters
    merged_clusters: pandas.DataFrame
        Merged clusters in single dataframe, re-indexed to avoid duplicate indices.
        To unambiguously identify clusters use the column 'uuids'
    """

    assert (len(clusters_list) == len(spikes_list)), 'clusters_list and spikes_list must have the same length'
    assert all([isinstance(s, dict) for s in spikes_list]), 'spikes_list must contain only dictionaries'
    assert all([isinstance(c, pd.DataFrame) for c in clusters_list]), 'clusters_list must contain only pd.DataFrames'

    merged_spikes = []
    merged_clusters = []
    cluster_max = 0
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes['clusters'] += cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)
    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()}
    # Sort spikes by spike time
    sort_idx = np.argsort(merged_spikes['times'], kind='stable')
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}

    return merged_spikes, merged_clusters


def load_trials_and_mask(
        one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', min_trial_len=None,
        max_trial_len=None, exclude_unbiased=False, exclude_nochoice=False, sess_loader=None,
        truncate_to_pass=True, saturation_intervals=None
):
    """
    Function to load all trials for a given session and create a mask to exclude all trials that have a reaction time
    shorter than min_rt or longer than max_rt or that have NaN for one of the specified events.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    eid: str
        A session UUID
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is 2. If None, don't apply.
    nan_exclude: list or 'default'
        List of trial events that cannot be NaN for a trial to be included. If set to 'default' the list is
        ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']
    min_trial_len: float or None
        Minimum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is None.
    max_trial_len: float or Nona
        Maximum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is None.
    exclude_unbiased: bool
        True to exclude trials that fall within the unbiased block at the beginning of session.
        Default is False.
    exclude_nochoice: bool
        True to exclude trials where the animal does not respond. Default is False.
    sess_loader: brainbox.io.one.SessionLoader or NoneType
        Optional SessionLoader object; if None, this object will be created internally
    truncate_to_pass: bool
        True to truncate sessions that don't pass performance on easy trials > 90 percent when all trials are used,
        but do pass when the first x > 400 trials are used. Default is True.
    saturation_intervals: str or list of str or None
         If str or list of str, the name of the interval(s) to be used to exclude trials if the ephys signal shows
         saturation in the interval(s). Default is None. Possible values are:
            saturation_stim_plus04
            saturation_feedback_plus04
            saturation_move_minus02
            saturation_stim_minus04_minus01
            saturation_stim_plus06
            saturation_stim_minus06_plus06
            saturation_stim_plus01

    Returns
    -------
    trials: pandas.DataFrame
        Trials table containing all trials for this session. If complete with columns:
        ['stimOff_times','goCueTrigger_times','feedbackType','contrastLeft','contrastRight','rewardVolume',
        'goCue_times','choice','feedback_times','stimOn_times','response_times','firstMovement_times',
        'probabilityLeft', 'intervals_0', 'intervals_1']
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

    if sess_loader is None:
        sess_loader = SessionLoader(one=one, eid=eid)

    if sess_loader.trials.empty:
        sess_loader.load_trials()

    # Truncate trials to pass performance on easy trials > 0.9
    good_enough = training.criterion_delay(
        n_trials=sess_loader.trials.shape[0],
        perf_easy=training.compute_performance_easy(sess_loader.trials),
    )
    if truncate_to_pass and not good_enough:
        n_trials = sess_loader.trials.shape[0]
        while not good_enough and n_trials > 400:
            n_trials -= 1
            sess_loader.trials = sess_loader.trials[:n_trials]
            good_enough = training.criterion_delay(
                n_trials=sess_loader.trials.shape[0],
                perf_easy=training.compute_performance_easy(sess_loader.trials),
            )
        if not good_enough:
            raise AssertionError('Session does not pass performance on easy trials > 0.9 for n_trials > 400')

    # Create a mask for trials to exclude
    # Remove trials that are outside the allowed reaction time range
    if min_rt is not None:
        query = f'(firstMovement_times - stimOn_times < {min_rt})'
    else:
        query = ''
    if max_rt is not None:
        query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    # Remove trials that are outside the allowed trial duration range
    if min_trial_len is not None:
        query += f' | (feedback_times - goCue_times < {min_trial_len})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    # Remove trials with nan in specified events
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    # Remove trials in unbiased block at beginning
    if exclude_unbiased:
        query += ' | (probabilityLeft == 0.5)'
    # Remove trials where animal does not respond
    if exclude_nochoice:
        query += ' | (choice == 0)'
    # If min_rt was None we have to clean up the string
    if min_rt is None:
        query = query[3:]

    # Create mask
    mask = ~sess_loader.trials.eval(query)

    # If saturation intervals are provided, download trials table to get information about saturation
    # Remove trials where the signal shows saturation in an interval of interest
    if saturation_intervals is not None:
        all_trials = pd.read_parquet(download_aggregate_tables(one, type='trials'))
        sess_trials = all_trials[all_trials['eid'] == eid]
        sess_trials.reset_index(inplace=True)
        assert sess_trials.shape[0] == sess_loader.trials.shape[0], 'Trials table does not match trials in session.'
        if isinstance(saturation_intervals, str):
            saturation_intervals = [saturation_intervals]
        for interval in saturation_intervals:
            mask[sess_trials[interval] == True] = False

    return sess_loader.trials, mask


def download_aggregate_tables(one, target_path=None, type='clusters', tag='2023_Q4_IBL_et_al_BWM_2', overwrite=False):
    """
    Function to download the aggregated clusters information associated with the given data release tag from AWS.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to database.
    target_path: str or pathlib.Path
        Directory to which clusters.pqt should be downloaded. If None, downloads to one.cache_dir/bwm_tables
    type: {'clusters', 'trials'}
        Which type of aggregate table to load, clusters or trials table.
    tag: str
        Tag for which to download the clusters table. Default is '2023_Q4_IBL_et_al_BWM_2''.
    overwrite : bool
        If True, will re-download files even if file exists locally and file sizes match.

    Returns
    -------
    agg_path: pathlib.Path
        Path to the downloaded aggregate
    """

    if target_path is None:
        target_path = Path(one.cache_dir).joinpath('bwm_tables')
        target_path.mkdir(exist_ok=True)
    else:
        assert target_path.exists(), 'The target_path you passed does not exist.'

    agg_path = target_path.joinpath(f'{type}.pqt')
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_file(f"aggregates/{tag}/{type}.pqt", agg_path, s3=s3,
                         bucket_name=bucket_name, overwrite=overwrite)

    if not agg_path.exists():
        print(f'Downloading of {type} table failed.')
        return
    return agg_path


def filter_units_region(eids, clusters_table=None, one=None, mapping='Beryl', min_qc=1., min_units_sessions=(5, 2)):
    """
    Filter to retain only units that satisfy certain region based criteria.

    Parameters
    ----------
    eids: list or pandas.Series
        List of session UUIDs to include at start.
    clusters_table: str or pathlib.Path
        Absolute path to clusters table to be used for filtering. If None, requires to provide one.api.ONE instance
        to download the latest version.
    mapping: str
        Mapping from atlas id to brain region acronym to be applied. Default is 'Beryl'.
    one: one.api.ONE
        Instance to be used to connect to download clusters_table if this is not explicitly provided.
    min_qc: float or None
        Minimum QC label for a spike sorted unit to be retained.
        Default is 1. If None, criterion is not applied.
    min_units_sessions: tuple or None
        If tuple, the first entry is the minimum of units per session per region for a session to be retained, the
        second entry is the minimum number of those sessions per region for a region to be retained.
        Default is (5, 2). If None, criterion is not applied

    Returns
    -------
    regions_df: pandas.DataFrame
        Dataframe of units that survive region based criteria.
    """

    if not any([min_qc, min_units_sessions]):
        print('No criteria selected. Aborting.')
        return

    if clusters_table is None:
        if one is None:
            print(f'You either need to provide a path to clusters_table or an instance of one.api.ONE to '
                  f'download clusters_table.')
            return
        else:
            clusters_table = download_aggregate_tables(one, type='clusters')
    clus_df = pd.read_parquet(clusters_table)

    # Only consider given pids
    clus_df = clus_df.loc[clus_df['eid'].isin(eids)]
    diff = set(eids).difference(set(clus_df['eid']))
    if len(diff) != 0:
        print('WARNING: Not all eids in bwm_df are found in cluster table.')

    # Only consider units that pass min_qc
    if min_qc:
        clus_df = clus_df.loc[clus_df['label'] >= min_qc]

    # Add region acronyms column and remove root and void regions
    br = BrainRegions()
    clus_df[f'{mapping}'] = br.id2acronym(clus_df['atlas_id'], mapping=f'{mapping}')
    clus_df = clus_df.loc[~clus_df[f'{mapping}'].isin(['void', 'root'])]

    # Group by regions and filter for sessions per region
    if min_units_sessions:
        units_count = clus_df.groupby([f'{mapping}', 'eid']).aggregate(
            n_units=pd.NamedAgg(column='cluster_id', aggfunc='count'),
        )
        # Only keep sessions with at least min_units_sessions[0] units
        units_count = units_count[units_count['n_units'] >= min_units_sessions[0]]
        # Only keep regions with at least min_units_sessions[1] sessions left
        units_count = units_count.reset_index(level=['eid'])
        region_df = units_count.groupby([f'{mapping}']).aggregate(
            n_sessions=pd.NamedAgg(column='eid', aggfunc='count'),
        )
        region_df = region_df[region_df['n_sessions'] >= min_units_sessions[1]]
        # Merge back to get the eids and clusters
        region_session_df = pd.merge(region_df, units_count, on=f'{mapping}', how='left')
        region_session_df = region_session_df.reset_index(level=[f'{mapping}'])
        region_session_df.drop(labels=['n_sessions', 'n_units'], axis=1, inplace=True)
        clus_df = pd.merge(region_session_df, clus_df, on=['eid', f'{mapping}'], how='left')

    # Reset index
    clus_df.reset_index(inplace=True, drop=True)

    return clus_df


def filter_sessions(eids, trials_table, bwm_include=True, min_errors=3, min_trials=None, saturation_intervals=None):
    """
    Filters eids for sessions that pass certain criteria.
    The function first loads an aggregate of all trials for the brain wide map dataset
     that contains already pre-computed acceptance critera


    Parameters
    ----------
    eids: list or pandas.Series
        Session ids to map to regions. Typically, the 'eid' column of the bwm_df returned by bwm_query.
        Note that these eids must be represented in trials_table to be considered for the filter.
    trials_table: str or pathlib.Path
        Absolute path to trials table to be used for filtering.
    bwm_include: bool
        Whether to filter for BWM inclusion criteria (see defaults of function load_trials_and_mask()). Default is True.
    min_errors: int or None
        Minimum number of error trials after other criteria are applied. Default is 3.
    min_trials: int or None
        Minimum number of trials that pass default criteria (see load_trials_and_mask()) for a session to be retained.
        Default is None, i.e. not applied
    saturation_intervals: str or list of str or None
         If str or list of str, the name of the interval(s) to be used to exclude trials if the ephys signal shows
         saturation in the interval(s). Default is None. Possible values are:
            saturation_stim_plus04
            saturation_feedback_plus04
            saturation_move_minus02
            saturation_stim_minus04_minus01
            saturation_stim_plus06
            saturation_stim_minus06_plus06
            saturation_stim_plus01

    Returns
    -------
    eids: pandas.Series
        Session ids that pass the criteria
    """

    # Load trials table
    trials_df = pd.read_parquet(trials_table)

    # Keep only eids
    trials_df = trials_df.loc[trials_df['eid'].isin(eids)]

    # Aggregate and filter
    if bwm_include:
        trials_df = trials_df[trials_df['bwm_include']]

    if saturation_intervals is not None:
        if isinstance(saturation_intervals, str):
            saturation_intervals = [saturation_intervals]
        for interval in saturation_intervals:
            trials_df = trials_df[~trials_df[interval]]

    trials_agg = trials_df.groupby('eid').aggregate(
        n_trials=pd.NamedAgg(column='eid', aggfunc='count'),
        n_error=pd.NamedAgg(column='feedbackType', aggfunc=lambda x: (x == -1).sum()),
    )
    if min_trials:
        trials_agg = trials_agg.loc[trials_agg['n_trials'] >= min_trials]
    if min_errors:
        trials_agg = trials_agg.loc[trials_agg['n_error'] >= min_errors]

    return trials_agg.index.to_list()


def bwm_units(one=None, freeze='2023_12_bwm_release', rt_range=(0.08, 0.2), min_errors=3,
              min_qc=1., min_units_sessions=(10, 2)):
    """
    Creates a dataframe with units that pass the current BWM inclusion criteria.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database.
    freeze: {None, 2022_10_initial, 2022_10_update, 2022_10_bwm_release, 2023_12_bwm_release}
        Default is 2023_12_bwm_release. If None, the database is queried for the current set of pids satisfying the
        criteria. If a string is specified, a fixed set of eids and pids is returned instead of querying the database.
    rt_range: tuple
        Admissible range of trial length measured by goCue_time (start) and feedback_time (end).
    min_errors: int or None
        Minimum number of error trials per session after other criteria are applied. Default is 3.
    min_qc: float
        Minimum quality criterion for a unit to be considered. Default is 1.
    min_units_sessions: tuple or None
        If tuple, the first entry is the minimum of units per session per region for a session to be retained, the
        second entry is the minimum number of those sessions per region for a region to be retained.
        Default is (10, 2). If None, criterion is not applied

    Returns
    -------
    unit_df: pandas.DataFrame
        Dataframe with units that pass the current BWM inclusion criteria.
    """

    # Get sessions and probes
    bwm_df = bwm_query(freeze=freeze)

    # Filter sessions on reaction time, no NaN in critical trial events (both implemented as bwm_include)
    # and min_errors per session
    trials_table = download_aggregate_tables(one, type='trials')
    if rt_range != (0.08, 0.2):
        raise NotImplementedError("Currently this function is only implemented for the default reaction time range of"
                                  "0.08 to 0.2 seconds. Please talk to a developer if you need to change this. ")
    eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, bwm_include=True, min_errors=min_errors)

    # Filter clusters on min_qc, min_units_region and min_sessions_region
    clusters_table = download_aggregate_tables(one, type='clusters')
    unit_df = filter_units_region(eids, clusters_table=clusters_table, mapping='Beryl', min_qc=min_qc,
                                  min_units_sessions=min_units_sessions)

    return unit_df


def filter_video_data(one, eids, camera='left', min_video_qc='FAIL', min_dlc_qc='FAIL'):
    """
    Filters sessions for which video and/or DLC data passes a given QC threshold for a selected camera.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database.
    eids: list or str
        List of session UUIDs to filter.
    camera: {'left', 'right', 'body'}
        Camera for which to filter QC. Default is 'left'.
    min_video_qc: {'CRITICAL', 'FAIL', 'WARNING', 'PASS', 'NOT_SET'} or None
        Minimum video QC threshold for a session to be retained. Default is 'FAIL'.
    min_dlc_qc: {'CRITICAL', 'FAIL', 'WARNING', 'PASS', 'NOT_SET'} or None
        Minimum dlc QC threshold for a session to be retained. Default is 'FAIL'.

    Returns
    -------
    list:
        List of session UUIDs that pass both indicated QC thresholds for the selected camera.

    Notes
    -----
    For the thresholds, note that 'NOT_SET' < 'PASS' < 'WARNING' < 'FAIL' < 'CRITICAL'
    If a min_video_qc or min_dlc_qc is set to None, all sessions are retained for that criterion.
    The intersection of sessions passing both criteria is returned.

    """

    # Check inputs
    if isinstance(eids, str):
        eids = [eids]
    assert isinstance(eids, list), 'eids must be a list of session uuids'
    assert min_video_qc in list(spec.QC._member_names_) + [None], f'{min_video_qc} is not a valid value for min_video_qc '
    assert min_dlc_qc in list(spec.QC._member_names_) + [None], f'{min_dlc_qc} is not a valid value for min_dlc_qc '
    assert min_video_qc or min_dlc_qc, 'At least one of min_video_qc or min_dlc_qc must be set'

    # Load QC json from cache and restrict to desired sessions
    with open(one.cache_dir.joinpath('QC.json'), 'r') as f:
        qc_cache = json.load(f)
    qc_cache = {qc['eid']: qc['extended_qc'] for qc in qc_cache if qc['eid'] in eids}
    assert set(list(qc_cache.keys())) == set(eids), 'Not all eids found in cached QC.json'

    # Passing video
    if min_video_qc is None:
        passing_vid = eids
    else:
        passing_vid = [
            k for k, v in qc_cache.items() if
            f'video{camera.capitalize()}' in v.keys() and
            spec.QC[v[f'video{camera.capitalize()}']].value <= spec.QC[min_video_qc].value
        ]

    # Passing dlc
    if min_dlc_qc is None:
        passing_dlc = eids
    else:
        passing_dlc = [
            k for k, v in qc_cache.items() if
            f'dlc{camera.capitalize()}' in v.keys() and
            spec.QC[v[f'dlc{camera.capitalize()}']].value <= spec.QC[min_dlc_qc].value
        ]

    # Combine
    passing = list(set(passing_vid).intersection(set(passing_dlc)))

    return passing
