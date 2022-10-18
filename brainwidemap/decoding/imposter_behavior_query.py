from pathlib import Path
import pandas as pd
from one.api import ONE

fname = Path(__file__).resolve().parent.joinpath('imposter_behavior_sessions.pqt')

one = ONE(mode='remote')
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

df = pd.DataFrame(columns=['eid'], data=eids)
df.to_parquet(fname)
print(f'Saved list of eids to {fname}')
