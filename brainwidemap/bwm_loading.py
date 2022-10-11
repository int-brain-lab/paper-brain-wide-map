from dateutil import parser
import numpy as np
import pandas as pd
from pathlib import Path

from iblutil.numerical import ismember
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from ibllib.atlas.regions import BrainRegions
from one.remote import aws


def bwm_query(one=None, alignment_resolved=True, return_details=False, freeze='2022_10_initial'):
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
    freeze: {None, 2022_10_initial}
        Default is 2022_10_initial. If None, the database is queried for the current set of pids satisfying the
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

        fixtures_path = Path(__file__).parent.joinpath('fixtures')
        freeze_file = fixtures_path.joinpath(f'{freeze}.csv')
        assert freeze_file.exists(), f'{freeze} does not seem to be a valid freeze.'

        print(f'Loading from fixtures/{freeze}.csv')
        bwm_df = pd.read_csv(freeze_file, index_col=0)
        bwm_df['date'] = [parser.parse(i).date() for i in bwm_df['date']]

        return bwm_df

    # Else, query the database
    assert one is not None, 'If freeze=None, you need to pass an instance of one.api.ONE'
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


def load_good_units(one, pid, compute_metrics=False, **kwargs):
    """
    Function to load the cluster information and spike trains for clusters that pass all quality metrics.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
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
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics).to_df()
    iok = clusters_labeled['label'] == 1
    good_clusters = clusters_labeled[iok]

    spike_idx, ib = ismember(spikes['clusters'], good_clusters.index)
    good_clusters.reset_index(drop=True, inplace=True)
    # Filter spike trains for only good clusters
    good_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    good_spikes['clusters'] = good_clusters.index[ib].astype(np.int32)

    return good_spikes, good_clusters


def load_trials_and_mask(
        one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', min_trial_len=0.,
        max_trial_len=100., exclude_unbiased_trials=False, exclude_nochoice_trials=False,
        sess_loader=None):
    """
    Function to load all trials for a given session and create a mask to exclude all trials that have a reaction time
    shorter than min_rt or longer than max_rt or that have NaN for one of the specified events.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    eid: str
        A session UUID
    min_rt: float
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08.
    max_rt: float
        Maximum admissible reaction time in seconds for a trial to be included. Default is 2.
    nan_exclude: list or 'default'
        List of trial events that cannot be NaN for a trial to be included. If set to 'default' the list is
        ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']
    min_trial_len: float
        Minimum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is 0.
    max_trial_len: float
        Maximum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is 100.
    exclude_unbiased_trials: bool
        True to exclude trials that fall within the unbiased block at the beginning of session.
        Default is False.
    exclude_nochoice_trials: bool
        True to exclude trials where the animal does not respond. Default is False.
    sess_loader: brainbox.io.one.SessionLoader or NoneType
        Optional SessionLoader object; if None, this object will be created internally

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

    if sess_loader is None:
        sess_loader = SessionLoader(one, eid)

    if sess_loader.trials.shape[0] == 0:
        sess_loader.load_trials()

    # Create a mask for trials to exclude
    # Remove trials that are outside the allowed reaction time range
    query = f'(firstMovement_times - stimOn_times < {min_rt}) | (firstMovement_times - stimOn_times > {max_rt})'
    # Remove trials that are outside the allowed trial duration range
    query += f' | (feedback_times - goCue_times < {min_trial_len}) | (feedback_times - goCue_times > {max_trial_len})'
    # Remove trials with nan in specified events
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    # Remove trials in unbiased block at beginning
    if exclude_unbiased_trials:
        query += ' | (probabilityLeft == 0.5)'
    # Remove trials where animal does not respond
    if exclude_nochoice_trials:
        query += ' | (choice == 0)'

    # Create mask
    mask = ~sess_loader.trials.eval(query)

    return sess_loader.trials, mask


def download_aggregate_tables(one, local_path=None, type='clusters', tag='2022_Q4_IBL_et_al_BWM', overwrite=False):
    """
    Function to download the aggregated clusters information associated with the given data release tag from AWS.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to database.
    local_path: str or pathlib.Path
        Directory to which clusters.pqt should be downloaded. If None, downloads to current working directory.
    type: {'clusters', 'trials'}
        Which type of aggregate table to load, clusters or trials table.
    tag: str
        Tag for which to download the clusters table. Default is '2022_Q4_IBL_et_al_BWM'.
    overwrite : bool
        If True, will re-download files even if file exists locally and file sizes match.

    Returns
    -------
    agg_path: pathlib.Path
        Path to the downloaded aggregate
    """

    local_path = Path.cwd() if local_path is None else Path(local_path)
    assert local_path.exists(), 'The local_path you passed does not exist.'

    agg_path = local_path.joinpath(f'{type}.pqt')
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_file(f"aggregates/{tag}/{type}.pqt", agg_path, s3=s3,
                         bucket_name=bucket_name, overwrite=overwrite)

    if not agg_path.exists():
        print(f'Downloading of {type} table failed.')
        return
    return agg_path


def filter_regions(pids, clusters_table=None, one=None, mapping='Beryl',
                   min_qc=1., min_units_region=10, min_probes_region=2, min_sessions_region=None):
    """
    Maps probes to regions and filters to retain only probe-regions pairs that satisfy certain criteria.

    Parameters
    ----------
    pids: list or pandas.Series
        Probe insertion ids to map to regions. Typically, the 'pid' column of the bwm_df returned by bwm_query.
        Note that these pids must be represented in clusters_table to be considered for the filter.
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
    min_units_region: int or None
        Minimum number of units per region for a region to be retained.
        Default is 10. If None, criterion is not applied
    min_probes_region: int or None
        Minimum number of probes per region for a region to be retained. Mutually exclusive with min_sessions_region.
        Default is 2. If None, criterion is not applied.
    min_sessions_region: int or None
        Minimum number of sessions per region for a region to be retained. Mutually exclusive with min_probes_region.
        Default is None, i.e. not applied

    Returns
    -------
    regions_df: pandas.DataFrame
        Dataframe of unique region-probe pairs, with columns ['{mapping}', 'pid', 'n_units', 'n_probes', 'n_sessions']
    """

    if not any([min_qc, min_units_region, min_probes_region, min_sessions_region]):
        print('No criteria selected. Aborting.')
        return

    if all([min_probes_region, min_sessions_region]):
        print('Only one of min_probes_region and min_session_region can be applied, the other must be None.')
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
    clus_df = clus_df.loc[clus_df['pid'].isin(pids)]
    diff = set(pids).difference(set(clus_df['pid']))
    if len(diff) != 0:
        print('WARNING: Not all pids in bwm_df are found in cluster table.')

    # Only consider units that pass min_qc
    if min_qc:
        clus_df = clus_df.loc[clus_df['label'] >= min_qc]

    # Add region acronyms column and remove root and void regions
    br = BrainRegions()
    clus_df[f'{mapping}'] = br.id2acronym(clus_df['atlas_id'], mapping=f'{mapping}')
    clus_df = clus_df.loc[~clus_df[f'{mapping}'].isin(['void', 'root'])]

    # Count units, probes and sessions per region
    regions_count = clus_df.groupby(f'{mapping}').aggregate(
        n_probes=pd.NamedAgg(column='pid', aggfunc='nunique'),
        n_units=pd.NamedAgg(column='cluster_id', aggfunc='count'),
        n_sessions=pd.NamedAgg(column='eid', aggfunc='nunique')
    )

    # Reset index
    regions_count.reset_index(inplace=True)

    clus_df = pd.merge(clus_df, regions_count, how='left', on=f'{mapping}')
    if min_units_region:
        clus_df = clus_df[clus_df.eval(f'n_units >= {min_units_region}')]
    if min_probes_region:
        clus_df = clus_df[clus_df.eval(f'n_probes >= {min_probes_region}')]
    if min_sessions_region:
        clus_df = clus_df[clus_df.eval(f'n_sessions >= {min_sessions_region}')]

    regions_df = clus_df.filter([f'{mapping}', 'pid', 'n_units', 'n_probes', 'n_sessions'])
    regions_df.drop_duplicates(inplace=True)
    regions_df.reset_index(inplace=True, drop=True)

    return regions_df


def filter_sessions(eids, trials_table=None, one=None, min_trials=200):
    """
    Filters eids for those that have fulfill certain criteria.

    Parameters
    ----------
    eids: list or pandas.Series
        Session ids to map to regions. Typically, the 'eid' column of the bwm_df returned by bwm_query.
        Note that these eids must be represented in clusters_table to be considered for the filter.
    trials_table: str or pathlib.Path
        Absolute path to trials table to be used for filtering. If None, requires to provide one.api.ONE instance
        to download the latest version. Required when using min_trials.
    one: one.api.ONE
        Instance to be used to connect to download clusters or trials table if these are not explicitly provided.
    min_trials: int or None
        Minimum number of trials that pass default criteria (see load_trials_and_mask()) for a session to be retained.
        Default is 200. If None, criterion is not applied

    Returns
    -------
    eids: pandas.Series
        Session ids that pass the criteria
    """

    if trials_table is None:
        if one is None:
            print(f'You either need to provide a path to trials_table or an instance of one.api.ONE to '
                  f'download trials_table.')
            return
        else:
            trials_table = download_aggregate_tables(one, type='trials')
    trials_df = pd.read_parquet(trials_table)

    # Keep only eids
    trials_df = trials_df.loc[trials_df['eid'].isin(eids)]

    # Count trials that pass bwm_qc
    pass_trials = trials_df.groupby('eid').aggregate(n_trials=pd.NamedAgg(column='bwm_include', aggfunc='sum'))
    pass_trials.reset_index(inplace=True)
    if min_trials:
        keep_eids = pass_trials.loc[pass_trials['n_trials'] >= min_trials]['eid'].unique()
    else:
        keep_eids = pass_trials['eid'].unique()

    return keep_eids