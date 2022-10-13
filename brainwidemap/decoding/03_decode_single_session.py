import sys
import numpy as np
import pandas as pd
from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap.bwm_loading import (
    bwm_query, load_good_units, load_trials_and_mask, filter_regions, filter_sessions,
    download_aggregate_tables)
from brainwidemap.decoding.settings import kwargs
from brainwidemap.decoding.functions.decoding import fit_eid
from brainwidemap.decoding.functions.process_targets import load_behavior
from brainwidemap.decoding.paths import BEH_MOD_PATH, FIT_PATH, IMPOSTER_SESSION_PATH

n_pseudo = kwargs['n_pseudo']
n_pseudo_per_job = kwargs['n_pseudo_per_job']

kwargs['add_to_saving_path'] = f"_binsize={1000 * kwargs['binsize']}_lags={kwargs['n_bins_lag']}_" \
                              f"mergedProbes_{kwargs['merged_probes']}"

# Take the argument given to this script and create index by subtracting 1
index = int(sys.argv[1]) - 1

# create the directories
kwargs['behfit_path'] = BEH_MOD_PATH
kwargs['neuralfit_path'] = FIT_PATH
kwargs['behfit_path'].mkdir(parents=True, exist_ok=True)
kwargs['neuralfit_path'].mkdir(parents=True, exist_ok=True)

# load the list of probe insertions and select probe
one = ONE(mode='local')
bwm_df = bwm_query()
cache_dir = one.cache_dir

# Download the latest clusters table, we use the same cache as above
clusters_table = download_aggregate_tables(one, type='clusters', target_path=cache_dir)
# Map probes to regions (here: Beryl mapping) and filter according to QC, number of units and
# probes per region
region_df = filter_regions(
    bwm_df['pid'], clusters_table=clusters_table, mapping='Beryl', min_qc=1, min_units_region=10,
    min_probes_region=2)

# Download the latest trials table and filter for sessions that have at least 200 trials fulfilling
# BWM criteria
trials_table = download_aggregate_tables(one, type='trials', target_path=cache_dir)
eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, min_trials=200)

# Remove probes and sessions based on those filters
bwm_df = bwm_df[bwm_df['pid'].isin(region_df['pid'].unique())]
bwm_df = bwm_df[bwm_df['eid'].isin(eids)]

# Not sure why we do this
# TODO: brandon wants to update
pid_idx = index % bwm_df.index.size
job_id = index // bwm_df.index.size
# n_jobs_per_eid = N_PSEUDO/N_PSEUDO_PER_JOB
# pid_idx = index // n_jobs_per_eid
# job_id = index % n_jobs_per_eid

#n_jobs_per_eid = N_PSEUDO/N_PSEUDO_PER_JOB
#pid_idx = index // n_jobs_per_eid
#job_id = index % n_jobs_per_eid

metadata = {
    'subject': bwm_df.iloc[pid_idx]['subject'],
    'eid': bwm_df.iloc[pid_idx]['eid'],
    'probe_name': bwm_df.iloc[pid_idx]['probe_name']
}

# load trials df
sess_loader = SessionLoader(one, metadata['eid'])
sess_loader.load_trials()

# create mask
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=metadata['eid'], sess_loader=sess_loader,
    min_rt=kwargs['min_rt'], max_rt=kwargs['max_rt'],
    min_trial_len=kwargs['min_len'], max_trial_len=kwargs['max_len'],
    exclude_nochoice=True, exclude_unbiased=kwargs['exclude_unbiased_trials'])

# load target data if necessary (will probably put this into a function eventually)
if kwargs['target'] in ['wheel-vel', 'wheel-speed', 'l-whisker-me', 'r-whisker-me']:

    # load target data
    dlc_dict = load_behavior(kwargs['target'], sess_loader)

    # load imposter sessions
    ephys_str = '_beforeRecording' if not kwargs['imposter_generate_from_ephys'] else ''
    filename = 'imposterSessions_%s%s.pqt' % (kwargs['target'], ephys_str)
    imposter_file = IMPOSTER_SESSION_PATH.joinpath(filename)
    kwargs['imposter_df'] = pd.read_parquet(imposter_file) if n_pseudo > 0 else None

else:
    dlc_dict = None
    kwargs['imposter_df'] = None

# Load spike sorting data and put it in a dictionary for now
spikes, clusters = load_good_units(
    one,
    bwm_df.iloc[pid_idx]['pid'],
    eid=bwm_df.iloc[pid_idx]['eid'],
    pname=bwm_df.iloc[pid_idx]['probe_name'],
    compute_metrics=True
)

neural_dict = {
    'spk_times': spikes['times'],
    'spk_clu': spikes['clusters'],
    'clu_regions': clusters['acronym'],
    'clu_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
    'clu_df': clusters
}

#TODO change to loop how desired
if (job_id + 1) * n_pseudo_per_job <= n_pseudo:
    print(f"pid_id: {pid_idx}")
    pseudo_ids = np.arange(job_id * n_pseudo_per_job, (job_id + 1) * n_pseudo_per_job) + 1
    if 1 in pseudo_ids:
        pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
    results_fit_eid = fit_eid(
        neural_dict=neural_dict,
        trials_df=trials_df,
        trials_mask=trials_mask,
        metadata=metadata,
        pseudo_ids=pseudo_ids,
        dlc_dict=dlc_dict,
        **kwargs)

print('Slurm job successful')
