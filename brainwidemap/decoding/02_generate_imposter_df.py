"""
Loop over eids and concatenate their information to form a block of data that we can sample from
to create null distributions of certain variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap import bwm_query
from brainwidemap.decoding.functions.process_targets import get_target_variable_in_df
from brainwidemap.decoding.settings import params
from brainwidemap.decoding.settings import RESULTS_DIR

# Prepare where to store imposter sessions
decoding_dir = RESULTS_DIR.joinpath('decoding')
decoding_dir.mkdir(exist_ok=True, parents=True)

filename = decoding_dir.joinpath(f"imposterSessions_{params['target']}.pqt")

# ephys sessions from one of 12 templates
one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')
bwm_df = bwm_query(freeze='2022_10_bwm_release')
eids = bwm_df['eid'].unique()

# basic columns that we want to keep
columns = [
    'probabilityLeft',
    'contrastRight',
    'feedbackType',
    'choice',
    'contrastLeft',
    'eid',
    'template_sess',
    'firstMovement_times',
    'goCue_times',
    'stimOn_times',
    'feedback_times',
]

# add additional columns if necessary
add_behavior_col = params['target'] not in ['pLeft', 'signcont', 'feedback', 'choice']
if add_behavior_col:    
    columns += [params['target']]

all_trialsdf = []
for i, eid in enumerate(eids):
    print('%i: %s' % (i, eid))
    try:
        sess_loader = SessionLoader(one=one, eid=eid)
        sess_loader.load_trials()
        trialsdf = sess_loader.trials
    except Exception as e:
        print('ERROR LOADING TRIALS DF')
        print(e)
        continue

    try:
        if add_behavior_col:
            trialsdf = get_target_variable_in_df(
                one=one, eid=eid, sess_loader=sess_loader, target=params['target'],
                align_time=params['align_time'], time_window=params['time_window'],
                binsize=params['binsize'], min_rt=params['min_rt'], max_rt=params['max_rt'],
                min_len=params['min_len'], max_len=params['max_len'],
                exclude_unbiased_trials=params['exclude_unbiased_trials'],
            )
        if trialsdf is None:
            continue
        else:
            trialsdf.loc[:, 'eid'] = eid
            trialsdf.loc[:, 'trial_id'] = trialsdf.index
            trialsdf.loc[:, 'template_sess'] = i
            all_trialsdf.append(trialsdf)
    except Exception as e:
        print('ERROR CREATING IMPOSTER SESSION')
        print(e)
        continue

all_trialsdf = pd.concat(all_trialsdf)

# save imposter sessions
all_trialsdf[columns].to_parquet(filename)
