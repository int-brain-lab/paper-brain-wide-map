"""
On the second part of the pipeline example, we loop over the dataframe
The analysis tools load the downloaded and cached version of the data.
"""
import copy
from datetime import datetime
import glob
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import sys
import yaml

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap.bwm_loading import bwm_query, load_good_units, load_trials_and_mask
from brainwidemap.decoding.paths import FIT_PATH, IMPOSTER_SESSION_PATH
from brainwidemap.decoding.settings import kwargs
from brainwidemap.decoding.functions.decoding import fit_eid


# Determines below and above which sessions should not be decoded
IMIN = 0
IMAX = 1

# add path info
kwargs['add_to_saving_path'] = '_binsize=%i_lags=%i_mergedProbes_%i' % (
    1000 * kwargs['binsize'], kwargs['n_bins_lag'], kwargs['merge_probes'],
)
kwargs['neuralfit_path'] = FIT_PATH
print(kwargs)

# set up logging
log_file = FIT_PATH.joinpath('decoding_%s_%s.log' % (kwargs['target'], kwargs['date']))
logging.basicConfig(
    filename=log_file, filemode='w', level=logging.DEBUG,
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add logging to console

# Instantiate ONE
one = ONE()

# Get the set of sessions
bwm_df = bwm_query()

# load imposter session df
if kwargs.get('n_pseudo', 0) > 0:
    ephys_str = '_beforeRecording' if not kwargs['imposter_generate_from_ephys'] else ''
    filename = 'imposterSessions_%s%s.pqt' % (kwargs['target'], ephys_str)
    imposter_path = IMPOSTER_SESSION_PATH.joinpath(filename)
    kwargs['imposter_df'] = pd.read_parquet(imposter_path)
else:
    kwargs['imposter_df'] = None

# loop over sessions: load data and decode

for i, eid in enumerate(bwm_df['eid'].unique()):
    # determine if we should proceed with decoding session
    if i < IMIN:
        continue
    if i >= IMAX:
        continue

    # construct metadata dict for now
    metadata = {
        'subject': bwm_df.iloc[i]['subject'],
        'eid': bwm_df.iloc[i]['eid'],
        'probe_name': bwm_df.iloc[i]['probe_name']
    }
    logging.log(logging.DEBUG, f"{i}, session: {eid}, subject: {metadata['subject']}")

    # load trials data (same for all probes)
    # add start and end times (for now)
    try:
        sess_loader = SessionLoader(one, metadata['eid'])
        sess_loader.load_trials()
        sess_loader.trials['trial_start'] = sess_loader.trials['stimOn_times'] - 0.6
        sess_loader.trials['trial_end'] = sess_loader.trials['stimOn_times'] + 0.6
    except Exception as e:
        logging.log(logging.CRITICAL, e)
        continue

    # Load whichever target data is required and put it into "dlc_dict" for now
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
        logging.info(e)
        dlc_dict = {'times': None, 'values': None, 'skip': True}

    # For each probe of this session, load neural data and run fit_eid
    for pid, pname in bwm_df[bwm_df['eid']==eid].filter(['pid', 'probe_name']).itertuples(index=False):
        # load neural data
        try:
            spikes, clusters = load_good_units(one, pid, eid=eid, pname=pname)
            neural_dict = {
                'spk_times': spikes['times'],
                'spk_clu': spikes['clusters'],
                'clu_regions': clusters['acronym'],
                'clu_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
                'clu_df': clusters
            }
        except Exception as e:
            logging.log(logging.CRITICAL, e)
            continue

        filenames = fit_eid(
            neural_dict=neural_dict,
            trials_df=sess_loader.trials,
            metadata=metadata,
            dlc_dict=dlc_dict,
            pseudo_ids=[-1],
            **copy.copy(kwargs)
        )
        print(filenames)
