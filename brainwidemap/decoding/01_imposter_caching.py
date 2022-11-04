"""
Loop over eids and load data needed to create impostor df
"""

import sys
import pandas as pd
from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap import bwm_query
from brainwidemap.decoding.settings import params

N_PARA = 500
PARAINDEX = int(sys.argv[1])-1

if params['imposter_generate_from_ephys']:
    # ephys sessions from from one of 12 templates
    one = ONE(mode='local')
    bwm_df = bwm_query()
    eids = bwm_df['eid'].unique()
else:
    # no template, no neural activity
    one = ONE(mode='remote')
    # eids = one.search(project='ibl_neuropixel_brainwide_01', task_protocol='biasedChoiceWorld')
    qc_pass = (
        '~session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
        '~session__extended_qc___task_response_feedback_delays__lt,0.9,'
        '~session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
        '~session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
        '~session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_reward_volumes__lt,0.9,'
        '~session__extended_qc___task_reward_volume_set__lt,0.9,'
        '~session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
        '~session__extended_qc___task_audio_pre_trial__lt,0.9,'
        '~session__extended_qc___task_wheel_integrity__lt,1.0,'
        'n_trials__gte,400'
    )
    sessions = list(one.alyx.rest(
        'sessions', 'list',
        task_protocol='biasedChoiceWorld',
        project='ibl_neuropixel_brainwide_01',
        dataset_types=['wheel.position'],
        performance_gte=70,
        django=qc_pass,
    ))
    eids = [s['id'] for s in sessions]

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
if (params['target'] != 'pLeft') \
        and (params['target'] != 'signcont') \
        and (params['target'] != 'feedback') \
        and (params['target'] != 'choice'):
    add_behavior_col = True
    columns += [params['target']]

all_trialsdf = []
for i, eid in enumerate(eids):
    if not (i%N_PARA == PARAINDEX):
        continue
    else:
        det = one.get_details(eid, full=True)
        print('%i: %s' % (i, eid))
        try:
            sess_loader = SessionLoader(one=one, eid=eid)
            sess_loader.load_trials()
            sess_loader.load_wheel()
        except Exception as e:
            print('ERROR LOADING TRIALS DF')
            print(e)
            continue
    
