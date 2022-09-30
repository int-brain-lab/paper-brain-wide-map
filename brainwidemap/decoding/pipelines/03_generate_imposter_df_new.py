"""
Loop over eids and concatenate their information to form a block of data that we can sample from
to create null distributions of certain variables.
"""

import pandas as pd
from pathlib import Path
import yaml

from one.api import ONE
import brainbox.io.one as bbone
from brainbox.task.closed_loop import generate_pseudo_session

from braindelphi.params import braindelphi_PATH, SETTINGS_PATH, FIT_PATH
from braindelphi.decoding.functions.utils import check_settings
from braindelphi.decoding.functions.process_targets import get_target_variable_in_df


# load settings as a dict
settings = yaml.safe_load(open(SETTINGS_PATH))
kwargs = check_settings(settings)

one = ONE()

if kwargs['imposter_generate_from_ephys']:
    # ephys sessions from from one of ? templates
    insdf = pd.read_parquet(braindelphi_PATH.joinpath('decoding', 'insertions.pqt'))
    insdf = insdf[insdf.spike_sorting != '']
    eids = insdf['eid'].unique()
else:
    # no template, no neural activity
    eids = one.search(
        project='ibl_neuropixel_brainwide_01',
        task_protocol='biasedChoiceWorld')

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
    'trial_start',

]
# add additional columns if necessary
add_behavior_col = False
if (kwargs['target'] != 'pLeft') \
        or (kwargs['target'] != 'signcont') \
        or (kwargs['target'] != 'feedback') \
        or (kwargs['target'] != 'choice'):
    add_behavior_col = True
    columns += [kwargs['target']]

all_trialsdf = []
for i, eid in enumerate(eids[:3]):

    det = one.get_details(eid, full=True)
    print('%i: %s' % (i, eid))
    # if 'ephys' in det['json']['PYBPOD_BOARD']:
    try:
        trialsdf = bbone.load_trials_df(eid, one=one)
    except Exception as e:
        print('ERROR LOADING TRIALS DF')
        print(e)
        continue

    # choose sessions that pass BWM criteria
    # - more than 400 trials
    # - better then 90% on highest contrasts trials
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
                trialsdf = get_target_variable_in_df(one, eid, **kwargs)
            if trialsdf is None:
                continue
            else:
                trialsdf['eid'] = eid
                trialsdf['trial_id'] = trialsdf.index
                trialsdf['template_sess'] = i
                all_trialsdf.append(trialsdf)
        except Exception as e:
            print('ERROR CREATING IMPOSTER SESSION')
            print(e)
            continue

all_trialsdf = pd.concat(all_trialsdf)

# save imposter sessions
ephys_str = '_beforeRecording' if not kwargs['imposter_generate_from_ephys'] else ''
filename = 'imposterSessions_%s%s.pqt' % (kwargs['target'], ephys_str)
all_trialsdf[columns].to_parquet(braindelphi_PATH.joinpath('decoding', filename))
