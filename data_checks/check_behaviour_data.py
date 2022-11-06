import pandas as pd
from pathlib import Path
from one.api import ONE
from one.alf.exceptions import ALFError
from brainwidemap import bwm_query
from brainbox.io.one import SessionLoader

__file__ = '/home/julia/workspace/int-brain-lab/paper-brain-wide-map/data_checks/check_behaviour_data.py'
trials_file = Path(__file__).parent.joinpath('check_trials_data.csv')
wheel_file = Path(__file__).parent.joinpath('check_wheel_data.csv')
behav_file = Path(__file__).parent.joinpath('check_behaviour_data.csv')

one = ONE()
bwm_df = bwm_query(one)

"""
--------
CRITERIA
--------
"""
# Queries to database based on which inclusion is decided
base_query = (
        'session__project__name__icontains,ibl_neuropixel_brainwide_01,'
        'session__json__IS_MOCK,False,'
        'session__qc__lt,50,'
        '~json__qc,CRITICAL,'
        'session__extended_qc__behavior,1,'
        'json__extended_qc__tracing_exists,True,'
        'json__extended_qc__alignment_resolved,True,')
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
    '~session__extended_qc___task_audio_pre_trial__lt,0.9')
marked_pass = (
    'session__extended_qc___experimenter_task,PASS')

# Sessions will be included if more than 200 trials fulfill these criteria
min_trials = 200
bwm_filter = (
    '(firstMovement_times - stimOn_times < 0.08) | '
    '(firstMovement_times - stimOn_times > 2) | '
    'stimOn_times.isnull() | '
    'choice.isnull() | '
    'feedback_times.isnull() | '
    'probabilityLeft.isnull() | '
    'firstMovement_times.isnull() | '
    'feedbackType.isnull()'
)

# Expected keys in trials df
trials_keys = ['stimOff_times', 'goCueTrigger_times', 'goCue_times', 'response_times', 'choice', 'stimOn_times',
               'contrastLeft', 'contrastRight', 'probabilityLeft', 'feedback_times', 'feedbackType', 'rewardVolume',
               'firstMovement_times', 'intervals_0', 'intervals_1']
# Expected keys in wheel df
wheel_keys = ['times', 'position', 'velocity', 'acceleration']


"""
-----------
COMPUTATION
-----------
"""

query_set = [i['session'] for i in one.alyx.rest('insertions', 'list', django=base_query + qc_pass)]
query_set.extend([i['session'] for i in one.alyx.rest('insertions', 'list', django=base_query + marked_pass)])
query_set = list(set(query_set))

all_trials = []
all_wheel = []
errors = []

for i, eid in enumerate(bwm_df['eid'].unique()):
    print(f"{i+1}/{bwm_df['eid'].nunique()}")
    sess_loader = SessionLoader(one, eid)

    # Run through individual trial checks
    trial_checks = [eid]
    trial_checks.append(eid in query_set)  # fulfills the db query criteria
    try:
        sess_loader.load_trials()
        sess_loader.trials.drop(columns=['intervals_bpod_0', 'intervals_bpod_1'], errors='ignore', inplace=True)
        trial_checks.append(True)  # Exists
        trial_checks.append(True)  # Timestamps match
        trial_checks.append(not sess_loader.trials.empty)  # Not emtpy
        trial_checks.append(set(sess_loader.trials.columns) == set(trials_keys))  # all columns
        trial_checks.append(all(~sess_loader.trials.isnull().values.all(axis=0)))  # not all nan for any column
        trial_checks.append((~sess_loader.trials.eval(bwm_filter)).sum() >= min_trials)  # bwm_filter passed for min_trials
    except ALFError as e:
        trial_checks.append(False)  # Doesn't exist
        errors.append((eid, e))
    except ValueError as e:
        trial_checks.append(True)  # Exist
        trial_checks.append(False)  # Timestamps mismatch
        errors.append((eid, e))
    all_trials.append(trial_checks)

    # Run through individual wheel checks
    wheel_checks = [eid]
    try:
        sess_loader.load_wheel()
        wheel_checks.append(True)  # Exists
        wheel_checks.append(True)  # Timestamps match
        wheel_checks.append(not sess_loader.wheel.empty)  # Not emtpy
        wheel_checks.append(set(sess_loader.wheel.columns) == set(wheel_keys))  # all columns
        wheel_checks.append(all(~sess_loader.wheel.isnull().values.all(axis=0)))  # not all nan for any column
    except ALFError as e:
        wheel_checks.append(False)  # Doesn't exist
        errors.append((eid, e))
    except ValueError as e:
        wheel_checks.append(True)  # Exist
        wheel_checks.append(False)  # Timestamps mismatch
        errors.append((eid, e))
    all_wheel.append(wheel_checks)


trials_df = pd.DataFrame(all_trials, columns=[
    'eid', 'qc_query', 'exists', 'times_match', 'not_empty', 'all_columns', 'not_all_nan', 'bwm_include'])
wheel_df = pd.DataFrame(all_wheel, columns=['eid', 'exists', 'times_match', 'not_empty', 'all_columns', 'not_nan'])

"""
---------
SIMPLIFY
---------
"""
# Combine into simplified behaviour dataframe
# For wheel require all tests to pass
wheel_total = wheel_df.all(axis=1).map({True: 'PASS', False: 'FAIL'})
wheel_total.name = 'Wheel'
# For trials if only bwm_include fails, put warning instead of FAIL
trials_total = trials_df.drop('bwm_include', axis=1).all(axis=1).map({True: 'PASS', False: 'FAIL'})
trials_total.loc[(trials_total == 'PASS') & (trials_df['bwm_include'] == False)] = 'WARNING'
trials_total.name = 'TrialEvents'

behav_df = pd.concat([trials_total, wheel_total], axis=1)

"""
------
OUTPUT
------
"""
print(errors)
print(f'Saving csvs')
trials_df.to_csv(trials_file)
wheel_df.to_csv(wheel_file)
behav_df.to_csv(behav_file)
