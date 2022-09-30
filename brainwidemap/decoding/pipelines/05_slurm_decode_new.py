import pandas as pd
import sys
from brainwidemap.decoding.settings import kwargs, N_PSEUDO_PER_JOB, N_PSEUDO
from brainwidemap.decoding.functions.decoding import fit_eid
import pandas as pd
import sys
from brainwidemap.decoding.settings import kwargs, N_PSEUDO_PER_JOB, N_PSEUDO
from brainwidemap.decoding.functions.decoding import fit_eid
import numpy as np
from one.api import ONE
from brainwidemap import bwm_query, load_good_units
from brainbox.io.one import SessionLoader
from pathlib import Path

T_BEF = 0.6
T_AFT = 0.6
BINWIDTH = 0.02

kwargs['behfit_path'] = Path('results/behavioral')
kwargs['neuralfit_path'] = Path('results/neural')
kwargs['behfit_path'].mkdir(parents=True, exist_ok=True)
kwargs['neuralfit_path'].mkdir(parents=True, exist_ok=True)

# Take the argument given to this script and create index by subtracting 1
try:
    index = int(sys.argv[1]) - 1
except:
    index = 32
    pass

# Load the list of probe insertions and select probe (not sure about this, why not just index directly)
one = ONE()
bwm_df = bwm_query(one)

pid_idx = index % bwm_df.index.size
job_id = index // bwm_df.index.size

metadata = {
    'subject': bwm_df.iloc[pid_idx]['subject'],
    'eid': bwm_df.iloc[pid_idx]['eid'],
    'probe_name': bwm_df.iloc[pid_idx]['probe_name']
}

# Load trials df and add start and end times (for now)
sess_loader = SessionLoader(one, metadata['eid'])
sess_loader.load_trials()
sess_loader.trials['trial_start'] = sess_loader.trials['stimOn_times'] - T_BEF
sess_loader.trials['trial_end'] = sess_loader.trials['stimOn_times'] + T_AFT

# Load spike sorting data and put it in a dictionary for now
spikes, clusters = load_good_units(one, bwm_df.iloc[pid_idx]['pid'], eid=bwm_df.iloc[pid_idx]['eid'],
                                   pname=bwm_df.iloc[pid_idx]['probe_name'])

neural_dict = {
    'spk_times': spikes['times'],
    'spk_clu': spikes['clusters'],
    'clu_regions': clusters['acronym'],
    'clu_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
    'clu_df': clusters
}

if (job_id + 1) * N_PSEUDO_PER_JOB <= N_PSEUDO:
    print(f"pid_id: {pid_idx}")
    pseudo_ids = np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
    if 1 in pseudo_ids:
        pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
    results_fit_eid = fit_eid(neural_dict=neural_dict, trials_df=sess_loader.trials, metadata=metadata,
                              pseudo_ids=pseudo_ids, **kwargs)
print('Slurm job successful')
