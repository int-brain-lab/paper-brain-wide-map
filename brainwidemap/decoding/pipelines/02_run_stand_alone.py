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

from braindelphi.utils_root import load_pickle_data
from braindelphi.params import CACHE_PATH, SETTINGS_PATH, FIT_PATH, IMPOSTER_SESSION_PATH
from braindelphi.pipelines.utils_common_pipelines import load_ephys, load_behavior
from braindelphi.decoding.functions.decoding import fit_eid
from braindelphi.decoding.functions.utils import check_settings


def load_metadata(neural_dtype_path_regex, date=None):
    '''
    Parameters
    ----------
    neural_dtype_path_regex
    date
    Returns
    -------
    the metadata neural_dtype from date if specified, else most recent
    '''
    neural_dtype_paths = glob.glob(neural_dtype_path_regex)
    neural_dtype_dates = [datetime.strptime(p.split('/')[-1].split('_')[0], '%Y-%m-%d')
                          for p in neural_dtype_paths]
    if date is None:
        path_id = np.argmax(neural_dtype_dates)
    else:
        path_id = np.argmax(np.array(neural_dtype_dates) == date)
    return pickle.load(open(neural_dtype_paths[path_id], 'rb')), neural_dtype_dates[path_id].strftime("%m-%d-%Y_%H:%M:%S")


# sessions to be excluded (by Olivier Winter)
excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load obect pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]


# load settings as a dict
settings = yaml.safe_load(open(SETTINGS_PATH))
kwargs = check_settings(settings)
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

# load insertion data
bwm_dict, _ = load_metadata(
    CACHE_PATH.joinpath('*_%s_metadata.pkl' % kwargs['neural_dtype']).as_posix())
bwm_df = bwm_dict['dataset_filenames']
eids = bwm_df['eid'].unique()

# load imposter session df
if kwargs.get('n_pseudo', 0) > 0:
    ephys_str = '_beforeRecording' if not kwargs['imposter_generate_from_ephys'] else ''
    filename = 'imposterSessions_%s%s.pqt' % (kwargs['target'], ephys_str)
    imposter_path = IMPOSTER_SESSION_PATH.joinpath(filename)
    kwargs['imposter_df'] = pd.read_parquet(imposter_path)
else:
    kwargs['imposter_df'] = None

# loop over sessions: load data and decode
one = ONE()
IMIN = 10
IMAX = 1000
for i, eid in enumerate(eids):

    # determine if we should proceed with decoding session
    if i < IMIN:
        continue
    if i >= IMAX:
        continue
    if eid in excludes:
        continue

    # use pre-computed pkl files
    curr_df = bwm_df[bwm_df['eid'] == eid]
    subject = curr_df.subject.iloc[0]

    # load behavioral data (same for all probes)
    filename = '%s_%s_regressors.pkl' % (kwargs['date'], kwargs['target'])
    beh_cache = Path(CACHE_PATH).joinpath('behavior', subject, eid, filename)
    # need to check if this exists since we're iterating through the neural data cache
    if beh_cache.exists():
        dlc_dict = load_pickle_data(beh_cache)
    else:
        logging.log(
            logging.DEBUG,
            f"{i}, session: {eid}, subject: {subject} - NO {kwargs['target']} DATA")
        continue

    if dlc_dict['skip']:
        continue

    logging.log(logging.DEBUG, f"{i}, session: {eid}, subject: {subject}")

    for pid_id in range(curr_df.index.size):

        pid = curr_df.iloc[pid_id]

        # load neural data
        try:
            metadata = pickle.load(open(pid.meta_file, 'rb'))
            regressors = pickle.load(open(pid.reg_file, 'rb'))
            if kwargs['neural_dtype'] == 'widefield':
                trials_df, neural_dict = regressors
            else:
                trials_df, neural_dict = regressors['trials_df'], regressors
        except Exception as e:
            logging.log(logging.CRITICAL, e)
            continue

        filenames = fit_eid(
            neural_dict=neural_dict,
            trials_df=trials_df,
            metadata=metadata,
            dlc_dict=dlc_dict,
            pseudo_ids=[-1],
            # pseudo_ids=1 + np.arange(kwargs['n_pseudo']),
            **copy.copy(kwargs)
        )
        print(filenames)
