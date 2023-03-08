from brainbox.io.one import SessionLoader
from one.api import ONE
import pandas as pd
import sys

from brainwidemap import (
    bwm_query, load_good_units, filter_regions, filter_sessions, download_aggregate_tables)
from brainwidemap.decoding.settings import RESULTS_DIR
from brainwidemap.decoding.settings import params


# save BiasedChoiceWorld sessions, no template, no neural activity
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
    
# save bwm dataframe of eids

one = ONE(base_url="https://openalyx.internationalbrainlab.org", mode='local')
bwm_df = bwm_query(freeze='2022_10_bwm_release')

# Download the latest clusters table, we use the same cache as above
clusters_table = download_aggregate_tables(one, type='clusters')
# Map probes to regions (here: Beryl mapping) and filter according to QC, number of units and probes per region
region_df = filter_regions(
    bwm_df['pid'], clusters_table=clusters_table, mapping='Beryl',
    min_qc=1, min_units_region=params['min_units'], min_probes_region=None, min_sessions_region=params['min_sess_per_reg'])
print('completed region_df')
# Download the latest trials table and filter for sessions that have at least 200 trials fulfilling BWM criteria
trials_table = download_aggregate_tables(one, type='trials',)
eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, min_trials=params['min_behav_trials'])
    
# Remove probes and sessions based on those filters
bwm_df = bwm_df[bwm_df['pid'].isin(region_df['pid'].unique())]
bwm_df = bwm_df[bwm_df['eid'].isin(eids)]
bwm_df.to_parquet(decoding_dir.joinpath('bwm_cache_sessions.pqt'))

