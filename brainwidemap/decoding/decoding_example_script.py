"""Example script that fits decoders for a single eid.

These are snippets of code taken from 04_decode_single_session.py to illustrate a simplified
pipeline. To run from the command line:

```
(iblenv) $ python decoding_example_script.py
```

"""

from brainbox.io.one import SessionLoader
import matplotlib.pyplot as plt
import numpy as np
from one.api import ONE
from pathlib import Path
import pickle

from brainwidemap.bwm_loading import bwm_query, load_good_units, load_trials_and_mask
from brainwidemap.decoding.functions.decoding import fit_eid
from brainwidemap.decoding.functions.utils import get_save_path
from brainwidemap.decoding.settings_template import params


# connect to server
one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    password='international',
    silent=True)

"""
--------------------------------
User input
--------------------------------
"""
# where results are saved
# results_dir = Path('/media/mattw/ibl/tmp')
results_dir = Path().home().joinpath('bwm_decoding_example')

# select example eid for decoding analysis
eid = 'b658bc7d-07cd-4203-8a25-7b16b549851b'

# perform decoding on original eid (-1 entry) and 5 pseudo-sessions
pseudo_ids = np.array([-1])

# select variable to decode
# targets other than 'pLeft' and 'signcont' (stimulus) require more processing to obtain
# the relevant null distributions that are not supported in this example; see
# 03_decode_single_session.py for more detailed information
params['target'] = 'pLeft'
params['tanh_transform'] = False  # only True for target=='signcont'

"""
--------------------------------
Update info from user selections
--------------------------------
"""
# update paths
params['behfit_path'] = results_dir.joinpath('decoding', 'results', 'behavioral')
params['behfit_path'].mkdir(parents=True, exist_ok=True)
params['neuralfit_path'] = results_dir.joinpath('decoding', 'results', 'neural')
params['neuralfit_path'].mkdir(parents=True, exist_ok=True)
params['add_to_saving_path'] = (
    f"_binsize={1000 * params['binsize']}_lags={params['n_bins_lag']}_mergedProbes_{False}")

# get other info from this eid
bwm_df = bwm_query(one, freeze='2022_10_bwm_release')
idx = bwm_df[bwm_df.eid == eid].index[1]  # take single probe
subject = bwm_df.iloc[idx]['subject']
pid = bwm_df.iloc[idx]['pid']
probe_name = bwm_df.iloc[idx]['probe_name']

"""
--------------------------------
Load data
--------------------------------
"""

# load trials df
sess_loader = SessionLoader(one, eid)
sess_loader.load_trials()

# create mask
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, sess_loader=sess_loader, min_rt=params['min_rt'], max_rt=params['max_rt'],
    min_trial_len=params['min_len'], max_trial_len=params['max_len'],
    exclude_nochoice=True, exclude_unbiased=params['exclude_unbiased_trials'])
params['trials_mask_diagnostics'] = [trials_mask]

# load target data if necessary
if params['target'] in ['wheel-vel', 'wheel-speed', 'l-whisker-me', 'r-whisker-me']:
    raise NotImplementedError(
        'see 04_decode_single_session.py for proper handling of wheel and dlc targets')
else:
    dlc_dict = None
    params['imposter_df'] = None

# Load spike sorting data
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
--------------------------------
Run decoding
--------------------------------
"""
# perform full nested xv decoding
# for pLeft, 5 pseudo-sessions, should take ~1 minute on a cpu
print(f'saving results to {results_dir}')
results_fit_eid = fit_eid(
    neural_dict=neural_dict,
    trials_df=trials_df,
    trials_mask=trials_mask,
    metadata=metadata,
    pseudo_ids=pseudo_ids,
    dlc_dict=dlc_dict,
    **params)

"""
--------------------------------
Load and plot results
--------------------------------
"""
region = 'CP'
save_path = get_save_path(
    -1, metadata['subject'], metadata['eid'], 'ephys',
    probe=metadata['probe_name'],
    region=region,
    output_path=params['neuralfit_path'],
    time_window=params['time_window'],
    date=params['date'],
    target=params['target'],
    add_to_saving_path=params['add_to_saving_path'],
)
# save_path = results_fit_eid[0]  # can also access results from list returned by fit function
results = pickle.load(open(save_path, 'rb'))
curr_fit = results['fit'][0]

# plotting
t = np.where(curr_fit['mask'])[0]
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, curr_fit['target'], label='True')
ax.plot(t, curr_fit['predictions_test'], label='Predicted')
ax.set_xlabel('Trial number')
ax.set_ylabel(params['target'])
ax.set_title('eid=%s\nsubject=%s\nregion=%s, probe=%s' % (
    results['eid'], results['subject'], results['region'][0], results['probe']))
ax.text(
    0.05, 0.9, '$R^2$=%1.2f' % curr_fit['scores_test_full'], transform=ax.transAxes, fontsize=12)
ax.legend(loc='upper right')
plt.tight_layout()
fig_path = save_path.parent.joinpath('example_result.png')
plt.savefig(fig_path)
plt.show()
print('find result figure at %s' % fig_path)
