import pandas as pd
from pathlib import Path
from one.api import ONE
from one.alf.exceptions import ALFError
from brainwidemap import bwm_query
from brainbox.io.one import SessionLoader

# Output files
trials_file = Path(__file__).parent.joinpath('check_trials_data.csv')
wheel_file = Path(__file__).parent.joinpath('check_wheel_data.csv')
video_file = Path(__file__).parent.joinpath('check_video_data.csv')
behav_file = Path(__file__).parent.joinpath('check_behaviour_data.csv')

# File created on SDSC (see check_video_sdsc.py)
sdsc_video_info = pd.read_csv(Path(__file__).parent.joinpath('video_status_sdsc.csv'))

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

# Manually checked video data
video_to_exclude = [
    'dd4da095-4a99-4bf3-9727-f735077dba66',
    'e5fae088-ed96-4d9b-82f9-dfd13c259d52',
    '4d8c7767-981c-4347-8e5e-5d5fffe38534',
    'cea755db-4eee-4138-bdd6-fc23a572f5a1',
    '19b44992-d527-4a12-8bda-aa11379cb08c',
    '8c2f7f4d-7346-42a4-a715-4d37a5208535',
    'f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1',
    'c728f6fd-58e2-448d-aefb-a72c637b604c',
    'af55d16f-0e31-4073-bdb5-26da54914aa2',
    'd832d9f7-c96a-4f63-8921-516ba4a7b61f',
    'dcceebe5-4589-44df-a1c1-9fa33e779727',
    '65f5c9b4-4440-48b9-b914-c593a5184a18',
    '4ddb8a95-788b-48d0-8a0a-66c7c796da96',
    'fa8ad50d-76f2-45fa-a52f-08fe3d942345',
    '8a3a0197-b40a-449f-be55-c00b23253bbf',
    '5455a21c-1be7-4cae-ae8e-8853a8d5f55e',
    '0c828385-6dd6-4842-a702-c5075f5f5e81',
    '19e66dc9-bf9f-430b-9d6a-acfa85de6fb7',
    '1e45d992-c356-40e1-9be1-a506d944896f',
]

video_to_include = [
    '45ef6691-7b80-4a43-bd1a-85fc00851ae8',
    '1928bf72-2002-46a6-8930-728420402e01',
    'f84045b0-ce09-4ace-9d11-5ea491620707',
    '91796ceb-e314-4859-9a1f-092f85cc846a',
    '6c6983ef-7383-4989-9183-32b1a300d17a',
    '671c7ea7-6726-4fbe-adeb-f89c2c8e489b',
    '6bb5da8f-6858-4fdd-96d9-c34b3b841593',
    '259927fd-7563-4b03-bc5d-17b4d0fa7a55',
    'e49d8ee7-24b9-416a-9d04-9be33b655f40',
    '5139ce2c-7d52-44bf-8129-692d61dd6403',
    '66d98e6e-bcd9-4e78-8fbb-636f7e808b29',
    'ebc9392c-1ecb-4b4b-a545-4e3d70d23611',
    '32d27583-56aa-4510-bc03-669036edad20',
    '4ef13091-1bc8-4f32-9619-107bdf48540c',
    '09394481-8dd2-4d5c-9327-f2753ede92d7',
    '952870e5-f2a7-4518-9e6d-71585460f6fe',
    '695a6073-eae0-49e0-bb0f-e9e57a9275b9',
    '03063955-2523-47bd-ae57-f7489dd40f15',
]


"""
-----------
COMPUTATION
-----------
"""

query_set = [i['session'] for i in one.alyx.rest('insertions', 'list', django=base_query + qc_pass)]
query_set.extend([i['session'] for i in one.alyx.rest('insertions', 'list', django=base_query + marked_pass)])
query_set = list(set(query_set))

# TRIALS
all_trials = []
trials_errors = []
for i, eid in enumerate(bwm_df['eid'].unique()):
    print(f"Trials {i+1}/{bwm_df['eid'].nunique()}")
    sess_loader = SessionLoader(one, eid)
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
        trials_errors.append((eid, e))
    except ValueError as e:
        trial_checks.append(True)  # Exist
        trial_checks.append(False)  # Timestamps mismatch
        trials_errors.append((eid, e))
    all_trials.append(trial_checks)
trials_df = pd.DataFrame(all_trials, columns=['eid', 'qc_query', 'exists', 'times_match', 'not_empty', 'all_columns',
                                              'not_all_nan', 'bwm_include'])
trials_df.to_csv(trials_file)

# WHEEL
all_wheel = []
wheel_errors = []
for i, eid in enumerate(bwm_df['eid'].unique()):
    print(f"Wheel {i + 1}/{bwm_df['eid'].nunique()}")
    sess_loader = SessionLoader(one, eid)
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
        wheel_errors.append((eid, e))
    except ValueError as e:
        wheel_checks.append(True)  # Exist
        wheel_checks.append(False)  # Timestamps mismatch
        wheel_errors.append((eid, e))
    all_wheel.append(wheel_checks)
wheel_df = pd.DataFrame(all_wheel, columns=['eid', 'exists', 'times_match', 'not_empty', 'all_columns', 'not_nan'])
wheel_df.to_csv(wheel_file)

# VIDEO
all_video = []
for i, eid in enumerate(bwm_df['eid'].unique()):
    print(f"Video {i + 1}/{bwm_df['eid'].nunique()}")
    session = one.alyx.rest('sessions', 'read', id=eid, no_cache=True)
    datasets = one.list_datasets(eid, collection='alf') + one.list_datasets(eid, collection='raw_video_data')
    mismatch_info = sdsc_video_info[sdsc_video_info['eid'] == eid]

    for label in ['left', 'right', 'body']:
        # Add data computed on SDSC
        data = {'eid': eid, 'label': label,
                'mp4': len(next((d for d in datasets if f'_iblrig_{label}Camera.raw.mp4' in d), '')) > 1,
                'timestamps': len(next((d for d in datasets if f'_ibl_{label}Camera.times.npy' in d), '')) > 1,
                'frames_match': mismatch_info[mismatch_info['label'] == label].iloc[0]['mismatch'] is False,
                'dlc': len(next((d for d in datasets if f'_ibl_{label}Camera.dlc.pqt' in d), '')) > 1,
                'features': len(next((d for d in datasets if f'_ibl_{label}Camera.features.pqt' in d), '')) > 1,
                'licks': len(next((d for d in datasets if f'licks.times.npy' in d), '')) > 1,
                'duration': mismatch_info[mismatch_info['label'] == label].iloc[0]['duration'],
                'videoQC': session['extended_qc'].get(f'video{label.capitalize()}', 'NOT_COMPUTED'),
                'dlcQC': session['extended_qc'].get(f'dlc{label.capitalize()}', 'NOT_COMPUTED')}
        # Add extended video and dlc QC info
        for vqc in ['focus', 'position', 'framerate', 'pin_state', 'brightness', 'resolution', 'timestamps',
                    'camera_times', 'file_headers', 'dropped_frames', 'wheel_alignment']:
            data[f'videoQC_{vqc}'] = session['extended_qc'].get(f'_video{label.capitalize()}_{vqc}', 'NOT_COMPUTED')
        for dqc in ['paw_far_nan', 'mean_in_bbox', 'paw_close_nan', 'pupil_blocked', 'trace_all_nan', 'lick_detection',
                    'pupil_diameter_snr', 'time_trace_length_match']:
            data[f'dlcQC_{dqc}'] = session['extended_qc'].get(f'_dlc{label.capitalize()}_{dqc}', 'NOT_COMPUTED')
        df = pd.DataFrame.from_dict([data])
        all_video.append(df)
video_df = pd.concat(all_video)
video_df.reset_index(drop=True, inplace=True)
video_df.to_csv(video_file)

"""
---------------------
SIMPLIFY AND COMBINE
---------------------
"""
# For wheel require all tests to pass
wheel_total = wheel_df.set_index('eid').all(axis=1).map({True: 'PASS', False: 'FAIL'})
wheel_total.name = 'Wheel'

# For trials if only bwm_include fails, put warning instead of FAIL
trials_total = trials_df.set_index('eid').drop('bwm_include', axis=1).all(axis=1).map({True: 'PASS', False: 'FAIL'})
trials_total.loc[(trials_total == 'PASS') & (trials_df['bwm_include'] == False)] = 'WARNING'
trials_total.name = 'TrialEvents'

# For video, clean up a bit
video_total = pd.DataFrame(index=pd.Series(video_df['eid'].unique(), name='eid'),
                           columns=['leftVideo', 'rightVideo', 'bodyVideo', 'leftDLC', 'rightDLC', 'bodyDLC'])

video_df['duration'] = video_df['duration'] > 100
for column in [c for c in video_df.columns if c.startswith('videoQC') or c.startswith('dlcQC')]:
    video_df[column] = [v[0] if isinstance(v, list) else v for v in video_df[column]]
video_df.set_index(['eid', 'label'], inplace=True)

for eid in video_total.index:
    for side in ['left', 'right', 'body']:
        # Check that dat is there and not super short, if there set to QC outcome
        video_total.loc[eid, f'{side}Video'] = (
            'MISSING' if not video_df.loc[eid, side].loc[['mp4', 'timestamps', 'duration']].all()
            else video_df.loc[eid, side]['videoQC']
        )
        # For DLC check that exists and QC doesn't fail
        video_total.loc[eid, f'{side}DLC'] = (
            'MISSING' if not video_df.loc[eid, side]['dlc'] else video_df.loc[eid, side]['dlcQC']
        )

        if video_total.loc[eid, f'{side}Video'] == 'FAIL':
            # If only framerate fails, set video to warning
            videoQC_keys = [k for k in video_df.keys() if k.startswith('videoQC_')]
            if video_df.loc[eid, side].loc[videoQC_keys].drop('videoQC_framerate').all():
                video_total.loc[eid, f'{side}Video'] = 'WARNING'
            # If position or brightness fail but dlc ok, set to warning
            if video_df.loc[eid, side].loc[videoQC_keys].drop(['videoQC_framerate',
                                                               'videoQC_position',
                                                               'videoQC_brightness']).all():
                if video_total.loc[eid, f'{side}DLC'] == 'PASS':
                    video_total.loc[eid, f'{side}Video'] = 'WARNING'
        # Double check that frames match
        if video_total.loc[eid, f'{side}Video'] == 'PASS' and not video_df.loc[eid, side]['frames_match']:
            video_total.loc[eid, f'{side}Video'] = 'FAIL'

# Make sure manually checked sessions are correctly set
video_total.loc[video_to_exclude, :] = "MISSING"
#video_total.loc[video_to_include, ['leftVideo', 'rightVideo', 'bodyVideo']]
video_total.loc['ebc9392c-1ecb-4b4b-a545-4e3d70d23611', 'bodyVideo'] = 'WARNING'

behav_df = pd.merge(trials_total, wheel_total, on='eid', how='outer')
behav_df = pd.merge(behav_df, video_total, on='eid', how='outer')
behav_df.columns = ['Session', 'TrialEvents', 'Wheel', 'leftVideo', 'rightVideo', 'bodyVideo',
                    'leftDLC', 'rightDLC', 'bodyDLC']
behav_df.to_csv(behav_file)

print('TRIALS:', trials_errors)
print('WHEEL:', wheel_errors)
