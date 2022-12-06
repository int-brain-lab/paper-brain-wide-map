import sys
import numpy as np
import pandas as pd

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap.bwm_loading import (
    bwm_query, load_good_units, load_trials_and_mask, filter_regions, filter_sessions,
    download_aggregate_tables, merge_probes)
from brainwidemap.decoding.functions.decoding import fit_eid
from brainwidemap.decoding.functions.process_targets import load_behavior
from brainwidemap.decoding.settings import params
from brainwidemap.decoding.settings import RESULTS_DIR

"""
--------------
Prepare inputs
--------------
"""

# Establish paths and filenames
params['behfit_path'] = RESULTS_DIR.joinpath('decoding', 'results', 'behavioral')
params['behfit_path'].mkdir(parents=True, exist_ok=True)
params['neuralfit_path'] = RESULTS_DIR.joinpath('decoding', 'results', 'neural')
params['neuralfit_path'].mkdir(parents=True, exist_ok=True)
params['add_to_saving_path'] = (f"_binsize={1000 * params['binsize']}_lags={params['n_bins_lag']}_"
                                f"mergedProbes_{params['merged_probes']}")
ephys_str = '_beforeRecording' if not params['imposter_generate_from_ephys'] else ''
imposter_file = RESULTS_DIR.joinpath('decoding', f"imposterSessions_{params['target']}{ephys_str}.pqt")

# Load ONE and the list of probe insertions and select probe(s)
one = ONE(mode='local')
bwm_df = bwm_query(freeze='2022_10_bwm_release')

# used for only running pipeline on a few subjects
#mysubs = [sys.argv[2], sys.argv[3], sys.argv[4]]
#bwm_df = bwm_df[bwm_df["subject"].isin(mysubs)] 

# Download the latest clusters table, we use the same cache as above
clusters_table = download_aggregate_tables(one, type='clusters')
# Map probes to regions (here: Beryl mapping) and filter according to QC, number of units and probes per region
region_df = filter_regions(
    bwm_df['pid'], clusters_table=clusters_table, mapping='Beryl',
    min_qc=1, min_units_region=10, min_probes_region=None, min_sessions_region=2)
print('completed region_df')
# Download the latest trials table and filter for sessions that have at least 200 trials fulfilling BWM criteria
trials_table = download_aggregate_tables(one, type='trials',)
eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, min_trials=200)

# Remove probes and sessions based on those filters
bwm_df = bwm_df[bwm_df['pid'].isin(region_df['pid'].unique())]
bwm_df = bwm_df[bwm_df['eid'].isin(eids)]

# When creating the slurm jobs the (one-based) ID of the job in the array is passed to this script as an input.
# Translate this into a zero-based index to select probes / sessions
slurm_job_id = int(sys.argv[1]) - 1
print(f'inside script, running slurm_job {slurm_job_id}')
# If there are more slurm jobs than probes/sessions, the same probe/session will be rerun with a new set of pseudo
# sessions. This job_repeat is 0 for the first round, 1 for the second round and so on
job_repeat = slurm_job_id // bwm_df.index.size
# Don't go any further if we are already at the end of pseudo sessions for this probe/session
if (job_repeat + 1) * params['n_pseudo_per_job'] > params['n_pseudo']: #TODO bb thinks this is wrong, bb flipped from <= to >, should be good now
    print('ended because job counter', job_repeat+1, params['n_pseudo_per_job'], params['n_pseudo'])
    exit()
# Otherwise use info to construct pseudo ids
pseudo_ids = np.arange(job_repeat * params['n_pseudo_per_job'], (job_repeat + 1) * params['n_pseudo_per_job']) + 1
if 1 in pseudo_ids:
    pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')

# When merging probes we are interested in eids, not pids
if params['merged_probes']:
    idx = slurm_job_id % bwm_df['eid'].nunique()  # Select the same eid for two consecutive jobs
    eid = bwm_df['eid'].unique()[idx]
    tmp_df = bwm_df.set_index(['eid', 'subject']).xs(eid, level='eid')
    subject = tmp_df.index[0]
    pids = tmp_df['pid'].to_list()  # Select all probes of this session
    probe_names = tmp_df['probe_name'].to_list()
    print(f"Running merged probes session: {eid}")
else:
    idx = slurm_job_id % bwm_df['pid'].nunique()  # Select the same pid for two consecutive jobs
    eid = bwm_df.iloc[idx]['eid']
    subject = bwm_df.iloc[idx]['subject']
    pid = bwm_df.iloc[idx]['pid']
    probe_name = bwm_df.iloc[idx]['probe_name']
    print(f"Running probe: {pid}")


"""
----------
Load data
----------
"""

# load trials df
sess_loader = SessionLoader(one, eid)
sess_loader.load_trials()

# create mask
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, sess_loader=sess_loader, min_rt=params['min_rt'], max_rt=params['max_rt'],
    min_trial_len=params['min_len'], max_trial_len=params['max_len'],
    exclude_nochoice=True, exclude_unbiased=params['exclude_unbiased_trials'])

# load target data if necessary (will probably put this into a function eventually)
if params['target'] in ['wheel-vel', 'wheel-speed', 'l-whisker-me', 'r-whisker-me']:
    # load target data
    dlc_dict = load_behavior(params['target'], sess_loader)
    # load imposter sessions
    params['imposter_df'] = pd.read_parquet(imposter_file) if params['n_pseudo'] > 0 else None
else:
    dlc_dict = None
    params['imposter_df'] = None

# Load spike sorting data
if params['merged_probes']:
    clusters_list = []
    spikes_list = []
    for pid, probe_name in zip(pids, probe_names):
        tmp_spikes, tmp_clusters = load_good_units(one, pid, eid=eid, pname=probe_name, compute_metrics=True)
        tmp_clusters['pid'] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)
else:
    spikes, clusters = load_good_units(one, pid, eid=eid, pname=probe_name, compute_metrics=True)

# Put everything into the input format fit_eid still expects at this point
neural_dict = {
    'spk_times': spikes['times'],
    'spk_clu': spikes['clusters'],
    'clu_regions': clusters['acronym'],
    'clu_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
    'clu_df': clusters
}
metadata = {
    'subject': subject,
    'eid': eid,
    'probe_name': probe_name
}


"""
-------------
Run decoding
-------------
"""

results_fit_eid = fit_eid(
    neural_dict=neural_dict,
    trials_df=trials_df,
    trials_mask=trials_mask,
    metadata=metadata,
    pseudo_ids=pseudo_ids,
    dlc_dict=dlc_dict,
    **params)
print('Job successful')
