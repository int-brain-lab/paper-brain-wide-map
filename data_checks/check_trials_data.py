import pandas as pd
from pathlib import Path
from one.api import ONE
from one.alf.exceptions import ALFError
from brainwidemap import bwm_query
from brainbox.io.one import SessionLoader

out_file = Path(__file__).parent.joinpath('check_trials_data.csv')

one = ONE()
bwm_df = bwm_query(one, freeze='2022_10_initial')

# Basic query for insertions and associated sessions to be considered
base_query = (
        'session__project__name__icontains,ibl_neuropixel_brainwide_01,'
        'session__json__IS_MOCK,False,'
        'session__qc__lt,50,'
        '~json__qc,CRITICAL,'
        'session__extended_qc__behavior,1,'
        'json__extended_qc__tracing_exists,True,'
        'json__extended_qc__alignment_resolved,True,'
)
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

query_set = [i['session'] for i in one.alyx.rest('insertions', 'list', django=base_query + qc_pass)]
query_set.extend([i['session'] for i in one.alyx.rest('insertions', 'list', django=base_query + marked_pass)])
query_set = list(set(query_set))

# Sessions will be included if more than 200 trials fulfill these criteria
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
min_trials = 200

# Expected keys in trials df
trials_keys = ['stimOff_times', 'goCueTrigger_times', 'goCue_times', 'response_times', 'choice', 'stimOn_times',
               'contrastLeft', 'contrastRight', 'probabilityLeft', 'feedback_times', 'feedbackType', 'rewardVolume',
               'firstMovement_times', 'intervals_0', 'intervals_1']

all_trials = []
errors = []

for i, eid in enumerate(bwm_df['eid'].unique()):
    print(f"{i+1}/{bwm_df['eid'].nunique()}")
    trial_checks = [eid]

    trial_checks.append(eid in query_set)  # fulfills the db query criteria
    sess_loader = SessionLoader(one, eid)
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

trials_df = pd.DataFrame(all_trials, columns=[
    'eid', 'qc_query', 'exists', 'times_match', 'not_empty', 'all_columns', 'not_all_nan', 'bwm_include'
])

print(errors)
print(f'Saving csv to {out_file}')
trials_df.to_csv(out_file)
