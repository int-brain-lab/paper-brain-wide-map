from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import (
    bwm_query, load_good_units, load_trials_and_mask, filter_regions, filter_sessions,
    download_aggregate_tables)
import sys
from brainwidemap.decoding.settings import RESULTS_DIR

# number of jobs which should be submited by a .sh file to parallelize data caching
# data is cached in N_PARA parallel jobs and para_index indicates which parallel job this is
N_PARA = 50
# para_index should be in [0, N_PARA-1]
para_index = int(sys.argv[1])-1

"""
-----------
USER INPUTS
-----------
"""
# Whether to remove data based on number of probes and units per region, good trials
# (see below for specifics)
regions_filter = True
trials_filter = True

min_trials = 0

# Whether to download wheel and whisker data as well
wheel_data = True
whisker_data = True
# You can choose the cache directory if you want it to be different from the default
cache_dir = None  # Path('/full/path/to/cache')

"""
----------
CACHE DATA
-----------
"""
# For caching, we use an online instance.
if cache_dir:
    one = ONE(cache_dir=cache_dir, 
              base_url='https://openalyx.internationalbrainlab.org',
              mode='remote')
else:
    one = ONE(base_url='https://openalyx.internationalbrainlab.org',
              mode='remote')
    cache_dir = one.cache_dir

BAD_PIDS = []  # can exclude a list of PIDS here, but currently not necessary
BAD_EIDS = [one.pid2eid(p)[0] for p in BAD_PIDS]

# Get the dataframe of all included sessions and probe insertions, use frozen query
bwm_df = bwm_query(freeze='2022_10_bwm_release')

if regions_filter:
    # Download the latest clusters table, we use the same cache as above
    clusters_table = download_aggregate_tables(one, type='clusters')
    # Map probes to regions (here: Beryl mapping) and filter according to QC, number of units and probes per region
    region_df = filter_regions(bwm_df['pid'], clusters_table=clusters_table, mapping='Beryl', min_qc=1,
                               min_units_region=10, min_probes_region=None, 
                               min_sessions_region=1)
    # Remove probes and sessions based on this filters
    bwm_df = bwm_df[bwm_df['pid'].isin(region_df['pid'].unique())]

if trials_filter:
    # Download the latest trials table and filter for sessions that have at least 200 trials fulfilling BWM criteria
    trials_table = download_aggregate_tables(one, type='trials')
    eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, min_trials=min_trials)
    bwm_df = bwm_df[bwm_df['eid'].isin(eids)]

# Download cluster information and spike trains for all good units.
for count, pid in enumerate(bwm_df['pid']):
    if not (count%N_PARA == para_index):
        continue
    elif pid in BAD_PIDS:
        continue
    else:
       print(f"Downloading spike sorting data for {pid}")
       try:
           spikes, clusters = load_good_units(one, pid, compute_metrics=False)
       except:
           print(f"Downloading spike sorting failed for pid {pid}, skipping")

# Download trials for all sessions
for count, eid in enumerate(bwm_df['eid']):
    if not (count%N_PARA == para_index):
        continue
    else: # Oddly not problem with errors here, dont need BAD_EIDS like wheel or spike sorting
        print(f"Downloading trials data for {eid}")
        try:
            sess_loader = SessionLoader(one, eid)
            sess_loader.load_trials()
        except:
            print(f"Downloading trials data failed for eid {eid}, skipping")

# Download wheel data for all sessions
if wheel_data:
    for count, eid in enumerate(bwm_df['eid']):
        if not (count % N_PARA == para_index):
            continue
        elif eid in BAD_EIDS:
            continue
        else:
            print(f"Downloading wheel data for {eid}")
            sess_loader = SessionLoader(one, eid)
            sess_loader.load_wheel()

# Download whisker data for all sessions
if whisker_data:
    me_err = []
    for count, eid in enumerate(bwm_df['eid']):
        if not (count % N_PARA == para_index):
            continue
        else:
            try:
                print(f"Downloading motion energy data for {eid}")
                sess_loader = SessionLoader(one, eid)
                sess_loader.load_motion_energy(views=['left', 'right'])
            except BaseException as e:
                print(eid, e)
                me_err.append((eid, e))

# save BiasedChoiceWorld sessions, no template, no neural activity
if para_index == 0: # only need one job to save these sessions

    # Prepare where to store imposter sessions eid list if using biased choice world
    decoding_dir = RESULTS_DIR.joinpath('decoding')
    decoding_dir.mkdir(exist_ok=True, parents=True)
    

    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    # eids = one.search(project='ibl_neuropixel_brainwide_01', task_protocol='biasedChoiceWorld')
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
        '~session__extended_qc___task_audio_pre_trial__lt,0.9,'
        '~session__extended_qc___task_wheel_integrity__lt,1.0,'
        'n_trials__gte,400'
    )
    sessions = one.alyx.rest(
        'sessions', 'list',
        task_protocol='biasedChoiceWorld',
        project='ibl_neuropixel_brainwide_01',
        dataset_types=['wheel.position'],
        performance_gte=70,
        django=qc_pass,
    )
    eids = [s['id'] for s in sessions]

    eid_df = pd.DataFrame(columns=['eid'], data=eids)    
    eid_df.to_parquet(decoding_dir.joinpath('imposter_behavior_sessions.pqt'))

