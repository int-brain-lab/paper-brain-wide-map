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

ephys_str = '_beforeRecording' if not params['imposter_generate_from_ephys'] else ''
filename = decoding_dir.joinpath(f"imposterSessions_{params['target']}{ephys_str}.pqt")

# Initiate ONE
one = ONE(mode='local')

# Decide which eids to use
if params['imposter_generate_from_ephys']:
    # ephys sessions from one of 12 templates
    bwm_df = bwm_query(freeze='2022_10_update')
    eids = bwm_df['eid'].unique()
else:
    # no template, no neural activity
    eids = pd.read_parquet(Path(__file__).resolve().parent.joinpath('imposter_behavior_sessions.pqt'))['eid']

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
add_behavior_col = False
if params['target'] not in ['pLeft', 'signcont', 'feedback', 'choice']:
    add_behavior_col = True
    columns += [params['target']]

all_trialsdf = []
for i, eid in enumerate(eids):
    if np.random.rand() > 1.1:
        continue
    else:
        print('%i: %s' % (i, eid))
        try:
            sess_loader = SessionLoader(one=one, eid=eid)
            sess_loader.load_trials()
            trialsdf = sess_loader.trials
        except Exception as e:
            print('ERROR LOADING TRIALS DF')
            print(e)
            continue

        # choose sessions that pass BWM criteria
        # - more than 400 trials
        # - better then 90% on highest contrasts trials
        # TODO: compute mask with load_trials_and_mask()?
        high_contrast_mask = (trialsdf.contrastLeft == 1) | (trialsdf.contrastRight == 1)
        frac_correct_hc = (trialsdf[high_contrast_mask].feedbackType == 1).mean()
        if (trialsdf.index.size > 400) \
            and (frac_correct_hc > 0.9) \
            and ((trialsdf.probabilityLeft == 0.5).sum() == 90) \
            and (trialsdf.probabilityLeft.values[0] == 0.5):
            try:
                if add_behavior_col:
                    trialsdf = get_target_variable_in_df(one, eid, sess_loader=sess_loader, **params)
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
