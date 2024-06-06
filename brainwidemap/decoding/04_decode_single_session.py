import sys
import numpy as np
import pandas as pd

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap.bwm_loading import load_good_units, load_trials_and_mask, merge_probes
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
imposter_file = RESULTS_DIR.joinpath('decoding', f"imposterSessions_{params['target']}.pqt")
bwm_session_file = RESULTS_DIR.joinpath('decoding', 'bwm_cache_sessions.pqt')

# Load ONE and bwm dataframe of sessions
one = ONE(base_url="https://openalyx.internationalbrainlab.org", mode='local')
bwm_df = pd.read_parquet(bwm_session_file)

# Feature to run a subset of BWM dataset filtering by subjects.
# To use this, add subject names to the end of the line that calls this script in 03_slurm*.sh.
# See 03_slurm*.sh for an examples which is commented out or read the `03_*` section of the README.
if len(sys.argv) > 2:
    print('using a subset of bwm dataset')
    #mysubs = [sys.argv[i] for i in range(2, len(sys.argv))]
    #bwm_df = bwm_df[bwm_df["subject"].isin(mysubs)]
    myeids = [sys.argv[i] for i in range(2, len(sys.argv))]
    bwm_df = bwm_df[bwm_df["eid"].isin(myeids)]

# the submitted slurm jobs need to cover a 2-d space of bwm pids (or eids)
# and pseudo sessions.  The two indicies in this space are "idx" and "job_repeat"
# where idx iterates over bwm pids (or eids if merged probes) and 
# job_repeat iterates over sets of n_speudo_per_job pseudo sessions until 
# all n_pseudo pseudo sessions are completed.  The number of job repeats needed
# is therefore ceil(n_pseudo / n_pseudo_per_job) 
# The slurm input (one-based index) is converted to a zero-based index called slurm_job_id
# which indexes over the entire 2-d space by snaking. slurm_job_id is then converted to
# idx and job_repeat as follows 
slurm_job_id = int(sys.argv[1]) - 1
bwm_index_size = bwm_df['eid'].nunique() if params['merged_probes'] else bwm_df['pid'].nunique()
idx = slurm_job_id % bwm_index_size
job_repeat = slurm_job_id // bwm_index_size
print(f'running slurm_job {slurm_job_id}, bwm index {idx}, and job repeat {job_repeat}')

# Use job_repeat to select pseudo_ids or exit if all pseudo_ids are completed
pseudo_ids = np.arange(job_repeat * params['n_pseudo_per_job'], (job_repeat + 1) * params['n_pseudo_per_job']) + 1
if 1 in pseudo_ids:
    pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
if pseudo_ids[0] > params['n_pseudo']:
    print(f"ended job because this job_repeat ({job_repeat}) does not include any pseudo sessions < {params['n_pseudo']}")
    exit()
if pseudo_ids[-1] > params['n_pseudo']:
    print(f"truncated job because this job_repeat ({job_repeat}) includes more than {params['n_pseudo']} pseudo sessions")
    pseudo_ids = pseudo_ids[pseudo_ids <= params['n_pseudo']]

# Use idx to select eid and pid
# When merging probes we are interested in eids, not pids
if params['merged_probes']:
    eid = bwm_df['eid'].unique()[idx]
    tmp_df = bwm_df.set_index(['eid', 'subject']).xs(eid, level='eid')
    subject = tmp_df.index[0]
    pids = tmp_df['pid'].to_list()  # Select all probes of this session
    probe_names = tmp_df['probe_name'].to_list()
    print(f"Running merged probes for session eid: {eid}")
else:
    eid = bwm_df.iloc[idx]['eid']
    subject = bwm_df.iloc[idx]['subject']
    pid = bwm_df.iloc[idx]['pid']
    probe_name = bwm_df.iloc[idx]['probe_name']
    print(f"Running probe pid: {pid}")


"""
----------
Load data
----------
"""

# load trials df
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()

# create mask
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, sess_loader=sess_loader, min_rt=params['min_rt'], max_rt=params['max_rt'],
    min_trial_len=params['min_len'], max_trial_len=params['max_len'],
    exclude_nochoice=True, exclude_unbiased=params['exclude_unbiased_trials'])
_, trials_mask_without_minrt = load_trials_and_mask(
    one=one, eid=eid, sess_loader=sess_loader, min_rt=None, max_rt=params['max_rt'],
    min_trial_len=params['min_len'], max_trial_len=params['max_len'],
    exclude_nochoice=True, exclude_unbiased=params['exclude_unbiased_trials'])
_, trials_mask_without_maxrt = load_trials_and_mask(
    one=one, eid=eid, sess_loader=sess_loader, min_rt=params['min_rt'], max_rt=None,
    min_trial_len=params['min_len'], max_trial_len=params['max_len'],
    exclude_nochoice=True, exclude_unbiased=params['exclude_unbiased_trials'])
_, trials_mask_withonly_nochoice = load_trials_and_mask(
    one=one, eid=eid, sess_loader=sess_loader, min_rt=None, max_rt=None,
    min_trial_len=None, max_trial_len=None,
    exclude_nochoice=True, exclude_unbiased=False)

params['trials_mask_diagnostics'] = [trials_mask,
                                     trials_mask_without_minrt,
                                     trials_mask_without_maxrt,
                                     trials_mask_withonly_nochoice]

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
        tmp_spikes, tmp_clusters = load_good_units(one, pid, eid=eid, pname=probe_name)
        tmp_clusters['pid'] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)
else:
    spikes, clusters = load_good_units(one, pid, eid=eid, pname=probe_name)

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
