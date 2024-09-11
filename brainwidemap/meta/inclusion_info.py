# https://github.com/int-brain-lab/paper-brain-wide-map/blob/cecac2295f2dd61fad5f5b51326702b566591baa/brainwidemap/meta/inclusion_info.py#L18
import pandas as pd
from pathlib import Path
from one.api import ONE
from iblatlas.regions import BrainRegions
from brainwidemap.bwm_loading import download_aggregate_tables, bwm_query, bwm_units, filter_sessions
import brainwidemap

regions = BrainRegions()
rt_range = (0.08, 0.2)
min_errors = 3
min_qc = 1.
min_units_sessions = (5, 2)

one = ONE(base_url='https://openalyx.internationalbrainlab.org')
one = ONE(base_url='https://alyx.internationalbrainlab.org')

file_trials_table = download_aggregate_tables(one, type='trials')
trials_df = pd.read_parquet(file_trials_table)
clus_df = pd.read_parquet(download_aggregate_tables(one, type='clusters'))
clus_df['Beryl'] = regions.id2acronym(clus_df['atlas_id'], mapping='Beryl')

outcomes = []

def get_stats_dict(name, clus_df, bwm_df=None):
    return {
        'name': name,
        'n_sessions': clus_df.eid.nunique() if bwm_df is None else bwm_df.eid.nunique(),
        'n_probes': clus_df.pid.nunique() if bwm_df is None else bwm_df.pid.nunique(),
        'n_units': clus_df.uuids.nunique(),
        'n_regions': clus_df.Beryl.nunique(),
    }


# %% All sessions and PIDs included in the data release
bwm_df = bwm_query(freeze='2023_12_bwm_release')
cluster_selection = clus_df['eid'].isin(bwm_df.eid.unique())
outcomes.append(get_stats_dict('Session and insertion QC', clus_df=clus_df.loc[cluster_selection], bwm_df=bwm_df))

# Filter trials on "bwm_include" meaning reaction time in range, no NaN in the following trial events
# 'stimOn_times', 'choice', 'feedback_times', 'probabilityLeft', 'firstMovement_times', 'feedbackType'
trials_df = trials_df.loc[trials_df['eid'].isin(bwm_df.eid.unique())]
trials_df = trials_df[trials_df['bwm_include']]
bwm_df = bwm_df.loc[bwm_df['eid'].isin(trials_df.eid.unique())]
cluster_selection = clus_df['eid'].isin(bwm_df.eid.unique())
outcomes.append(get_stats_dict('Session and insertion QC', clus_df=clus_df.loc[cluster_selection], bwm_df=bwm_df))


# %% Filter on minimum of 3 errors per session
trials_agg = trials_df.groupby('eid').aggregate(
    n_trials=pd.NamedAgg(column='eid', aggfunc='count'),
    n_error=pd.NamedAgg(column='feedbackType', aggfunc=lambda x: (x == -1).sum()),
)
trials_agg = trials_agg.loc[trials_agg['n_error'] >= min_errors]
eids = trials_agg.index.to_list()
eids_ = filter_sessions(bwm_df['eid'].unique(), trials_table=file_trials_table, bwm_include=True, min_errors=min_errors)
assert set(eids) == set(eids_)

bwm_df = bwm_df.loc[bwm_df['eid'].isin(eids)]
clus_df = clus_df.loc[clus_df['eid'].isin(bwm_df.eid.unique())]
outcomes.append(get_stats_dict('Minimum 3 error trials', clus_df=clus_df, bwm_df=bwm_df))

# %% Filter clusters on min_qc
clus_df = clus_df.loc[clus_df['label'] >= min_qc]
outcomes.append(get_stats_dict('Single unit QC', clus_df=clus_df))

# %% Remove void and root
clus_df = clus_df.loc[~clus_df[f'Beryl'].isin(['void', 'root'])]
outcomes.append(get_stats_dict('Gray matter regions', clus_df=clus_df))

# %% Filter for min 10 units per region
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
outcomes.append(get_stats_dict('Minimum 5 units per region', clus_df=clus_df))

units_df = bwm_units(one, min_units_sessions=(5, -1), enforce_version=False, min_qc=1)
assert units_df.shape[0] == clus_df.uuids.nunique()

# %% Filter min 2 sessions per region
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
outcomes.append(get_stats_dict('Minimum 2 sessions per region', clus_df=clus_df))
units_df = bwm_units(one, min_units_sessions=(5, 2), enforce_version=False, min_qc=1)
assert(set(clus_df.eid.unique()) == set(units_df.eid.unique()))
assert(set(clus_df.pid.unique()) == set(units_df.pid.unique()))


# %% Filter 20 neurons per region good units
import numpy as np
iregions = clus_df['Beryl'].value_counts() >= 20
clus_df = clus_df.loc[clus_df['Beryl'].isin(iregions[iregions].index)]
outcomes.append(get_stats_dict('Minimum 20 units per region', clus_df=clus_df))


# %%
outcome_df = pd.DataFrame(outcomes).set_index('name')
outcome_df.index.name = None
print(outcome_df)
outcome_df.to_csv(Path(brainwidemap.__file__).parent.joinpath('meta', 'inclusion_info.csv'))

#                                n_sessions  n_probes  n_units  n_regions
# Session and insertion QC              459       699   621733        281
# Session and insertion QC              459       699   621733        281
# Minimum 3 error trials                459       699   621733        281
# Single unit QC                        459       698    75708        268
# Gray matter regions                   459       698    65336        266
# Minimum 5 units per region            455       691    63357        245
# Minimum 2 sessions per region         454       690    62990        210
# Minimum 20 units per region           454       689    62857        201
