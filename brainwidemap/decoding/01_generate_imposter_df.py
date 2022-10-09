"""
Loop over eids and concatenate their information to form a block of data that we can sample from
to create null distributions of certain variables.
"""

import pandas as pd
from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainbox.task.closed_loop import generate_pseudo_session

# from braindelphi.params import braindelphi_PATH, SETTINGS_PATH, FIT_PATH
from brainwidemap import bwm_query
from brainwidemap.decoding.functions.process_targets import get_target_variable_in_df
from brainwidemap.decoding.paths import IMPOSTER_SESSION_PATH
from brainwidemap.decoding.settings import kwargs


if kwargs['imposter_generate_from_ephys']:
    # ephys sessions from from one of 12 templates
    one = ONE(mode='local')
    bwm_df = bwm_query()
    eids = bwm_df['eid'].unique()
else:
    # no template, no neural activity
    one = ONE(mode='local')
    eids = one.search(project='ibl_neuropixel_brainwide_01', task_protocol='biasedChoiceWorld')

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
if (kwargs['target'] != 'pLeft') \
        and (kwargs['target'] != 'signcont') \
        and (kwargs['target'] != 'feedback') \
        and (kwargs['target'] != 'choice'):
    add_behavior_col = True
    columns += [kwargs['target']]

all_trialsdf = []
for i, eid in enumerate(eids):
    if (i%10) > 0:
        continue
    det = one.get_details(eid, full=True)
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
            if kwargs['imposter_generate_fake']:
                trialsdf = generate_pseudo_session(trialsdf)
            if add_behavior_col:
                if kwargs['imposter_generate_fake']:
                    raise NotImplementedError
                trialsdf = get_target_variable_in_df(one, eid, trials_df=trialsdf, **kwargs)
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
ephys_str = '_beforeRecording' if not kwargs['imposter_generate_from_ephys'] else ''
filename = 'imposterSessions_%s%s.pqt' % (kwargs['target'], ephys_str)
all_trialsdf[columns].to_parquet(IMPOSTER_SESSION_PATH.joinpath(filename))
