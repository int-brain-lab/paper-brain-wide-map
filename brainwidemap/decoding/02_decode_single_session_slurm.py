import sys
import numpy as np
import pandas as pd
from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap.bwm_loading import bwm_query, load_good_units
from brainwidemap.decoding.settings import kwargs
from brainwidemap.decoding.functions.decoding import fit_eid
from brainwidemap.decoding.paths import BEH_MOD_PATH, FIT_PATH, IMPOSTER_SESSION_PATH

# user settings
results_dir = Path('/full/path/to/results')

n_pseudo = kwargs['n_pseudo']
n_pseudo_per_job = kwargs['n_pseudo_per_job']

kwargs['add_to_saving_path'] = f"_binsize={1000 * kwargs['binsize']}_lags={kwargs['n_bins_lag']}_" \
                               f"mergedProbes_{kwargs['merge_probes']}"

# Take the argument given to this script and create index by subtracting 1
try:
    index = int(sys.argv[1]) - 1
except:
    index = 32
    pass

# Create the directories
kwargs['behfit_path'] = BEH_MOD_PATH
kwargs['neuralfit_path'] = FIT_PATH
kwargs['behfit_path'].mkdir(parents=True, exist_ok=True)
kwargs['neuralfit_path'].mkdir(parents=True, exist_ok=True)

# Load the list of probe insertions and select probe
one = ONE(mode='local')
bwm_df = bwm_query()

# Not sure why we do this
pid_idx = index % bwm_df.index.size
job_id = index // bwm_df.index.size

metadata = {
    'subject': bwm_df.iloc[pid_idx]['subject'],
    'eid': bwm_df.iloc[pid_idx]['eid'],
    'probe_name': bwm_df.iloc[pid_idx]['probe_name']
}

# Load trials df
sess_loader = SessionLoader(one, metadata['eid'])
sess_loader.load_trials()

# Load target data if necessary (will probably put this into a function eventually)
if kwargs['target'] in ['wheel-vel', 'l-whisker-me', 'r-whisker-me']:

    # load target data
    try:
        if kwargs['target'] == 'wheel-vel':
            sess_loader.load_wheel()
            dlc_dict = {'times': sess_loader.wheel['times'], 'values': sess_loader.wheel['velocity']}
        elif kwargs['target'] == 'wheel-speed':
            sess_loader.load_wheel()
            dlc_dict = {'times': sess_loader.wheel['times'], 'values': np.abs(sess_loader.wheel['velocity'])}
        elif kwargs['target'] == 'l-whisker-me':
            sess_loader.load_motion_energy(views=['left'])
            dlc_dict = {'times': sess_loader.motion_energy['leftCamera']['times'],
                        'values': sess_loader.motion_energy['leftCamera']['whiskerMotionEnergy']}
        elif kwargs['target'] == 'r-whisker-me':
            sess_loader.load_motion_energy(views=['right'])
            dlc_dict = {'times': sess_loader.motion_energy['rightCamera']['times'],
                        'values': sess_loader.motion_energy['rightCamera']['whiskerMotionEnergy']}
        dlc_dict['skip': False]
    except BaseException as e:
        dlc_dict = {'times': None, 'values': None, 'skip': True}

    # Load imposter sessions
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

if (job_id + 1) * n_pseudo_per_job <= n_pseudo:
    print(f"pid_id: {pid_idx}")
    pseudo_ids = np.arange(job_id * n_pseudo_per_job, (job_id + 1) * n_pseudo_per_job) + 1
    if 1 in pseudo_ids:
        pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
    results_fit_eid = fit_eid(
        neural_dict=neural_dict,
        trials_df=sess_loader.trials,
        metadata=metadata,
        pseudo_ids=pseudo_ids,
        dlc_dict=dlc_dict,
        **kwargs)
print('Slurm job successful')
