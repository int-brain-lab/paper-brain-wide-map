import pandas as pd
from pathlib import Path
from one.api import ONE
from ibllib.atlas.regions import BrainRegions
from brainwidemap.bwm_loading import download_aggregate_tables, bwm_query, bwm_units

rt_range = (0.08, 0.2)
min_errors = 3
min_qc = 1.
min_units_sessions = (10, 2)

one = ONE(base_url='https://openalyx.internationalbrainlab.org')
trials_df = pd.read_parquet(download_aggregate_tables(one, type='trials'))
clus_df = pd.read_parquet(download_aggregate_tables(one, type='clusters'))

outcomes = []

# All sessions and PIDs included in the data release
bwm_df = bwm_query()
outcomes.append({
    'name': 'Released data',
    'n_sessions': bwm_df.eid.nunique(),
    'n_probes': bwm_df.pid.nunique(),
    'n_units': clus_df.loc[clus_df['eid'].isin(bwm_df.eid.unique())].uuids.nunique(),
})


# Filter trials on "bwm_include" meaning reaction time in range, no NaN in the following trial events
# 'stimOn_times', 'choice', 'feedback_times', 'probabilityLeft', 'firstMovement_times', 'feedbackType'
trials_df = trials_df.loc[trials_df['eid'].isin(bwm_df.eid.unique())]
trials_df = trials_df[trials_df['bwm_include']]
bwm_df = bwm_df.loc[bwm_df['eid'].isin(trials_df.eid.unique())]
outcomes.append({
    'name': 'Trials filter reaction time and hardware failure',
    'n_sessions': bwm_df.eid.nunique(),
    'n_probes': bwm_df.pid.nunique(),
    'n_units': clus_df.loc[clus_df['eid'].isin(bwm_df.eid.unique())].uuids.nunique(),
}
)

# Filter on minimum of 3 errors per session
trials_agg = trials_df.groupby('eid').aggregate(
    n_trials=pd.NamedAgg(column='eid', aggfunc='count'),
    n_error=pd.NamedAgg(column='feedbackType', aggfunc=lambda x: (x == -1).sum()),
)
trials_agg = trials_agg.loc[trials_agg['n_error'] >= 3]
eids = trials_agg.index.to_list()
bwm_df = bwm_df.loc[bwm_df['eid'].isin(eids)]
outcomes.append({
    'name': 'Minimum 3 errors per session',
    'n_sessions': bwm_df.eid.nunique(),
    'n_probes': bwm_df.pid.nunique(),
    'n_units': clus_df.loc[clus_df['eid'].isin(bwm_df.eid.unique())].uuids.nunique(),
})

# Filter clusters on min_qc
clus_df = clus_df.loc[clus_df['eid'].isin(bwm_df.eid.unique())]
clus_df = clus_df.loc[clus_df['label'] >= min_qc]
outcomes.append({
    'name': 'Unit QC',
    'n_sessions': clus_df.eid.nunique(),
    'n_probes': clus_df.pid.nunique(),
    'n_units': clus_df.uuids.nunique(),
})


# Remove void and root
br = BrainRegions()
clus_df['Beryl'] = br.id2acronym(clus_df['atlas_id'], mapping='Beryl')
clus_df = clus_df.loc[~clus_df[f'Beryl'].isin(['void', 'root'])]
outcomes.append({
    'name': 'Remove non-brain regions',
    'n_sessions': clus_df.eid.nunique(),
    'n_probes': clus_df.pid.nunique(),
    'n_units': clus_df.uuids.nunique(),
})

# Filter for min 10 units per region
units_count = clus_df.groupby(['Beryl', 'eid']).aggregate(
    n_units=pd.NamedAgg(column='cluster_id', aggfunc='count'),
)
units_count = units_count[units_count['n_units'] >= min_units_sessions[0]]
units_count = units_count.reset_index(level=['eid'])
region_df = units_count.groupby(['Beryl']).aggregate(
    n_sessions=pd.NamedAgg(column='eid', aggfunc='count'),
)
region_session_df = pd.merge(region_df, units_count, on=f'Beryl', how='left')
region_session_df = region_session_df.reset_index(level=[f'Beryl'])
region_session_df.drop(labels=['n_sessions', 'n_units'], axis=1, inplace=True)
clus_df = pd.merge(region_session_df, clus_df, on=['eid', f'Beryl'], how='left')
outcomes.append({
    'name': 'Minimum 10 units per region',
    'n_sessions': clus_df.eid.nunique(),
    'n_probes': clus_df.pid.nunique(),
    'n_units': clus_df.uuids.nunique(),
})

# Filter min 2 sessions per region
units_count = clus_df.groupby([f'Beryl', 'eid']).aggregate(
    n_units=pd.NamedAgg(column='cluster_id', aggfunc='count'),
)
units_count = units_count.reset_index(level=['eid'])
region_df = units_count.groupby([f'Beryl']).aggregate(
    n_sessions=pd.NamedAgg(column='eid', aggfunc='count'),
)
region_df = region_df[region_df['n_sessions'] >= min_units_sessions[1]]
region_session_df = pd.merge(region_df, units_count, on=f'Beryl', how='left')
region_session_df = region_session_df.reset_index(level=[f'Beryl'])
region_session_df.drop(labels=['n_sessions', 'n_units'], axis=1, inplace=True)
clus_df = pd.merge(region_session_df, clus_df, on=['eid', f'Beryl'], how='left')
outcomes.append({
    'name': 'Minimum 2 sessions per region',
    'n_sessions': clus_df.eid.nunique(),
    'n_probes': clus_df.pid.nunique(),
    'n_units': clus_df.uuids.nunique(),
})

# Double check that the common function gives the same
units_df = bwm_units(one)
assert(set(clus_df.eid.unique()) == set(units_df.eid.unique()))
assert(set(clus_df.pid.unique()) == set(units_df.pid.unique()))


outcome_df = pd.DataFrame(outcomes).set_index('name')
outcome_df.index.name = None
outcome_df.to_csv('excluded.csv')