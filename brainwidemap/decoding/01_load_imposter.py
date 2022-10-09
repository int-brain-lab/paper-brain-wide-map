"""
Loop over eids and load data needed to create impostor df
"""

import sys
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

N_PARA = 500
PARAINDEX = int(sys.argv[1])-1

if kwargs['imposter_generate_from_ephys']:
    # ephys sessions from from one of 12 templates
    one = ONE(mode='local')
    bwm_df = bwm_query()
    eids = bwm_df['eid'].unique()
else:
    # no template, no neural activity
    one = ONE(mode='remote')
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
    if not (i%N_PARA == PARAINDEX):
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
