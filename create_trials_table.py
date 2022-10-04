import pandas as pd
from one.api import ONE
from brainwidemap import bwm_query
from brainbox.io.one import SessionLoader

local_path = '/home/julia/data/2022_Q4_IBL_et_al_BWM/trials.pqt'

one = ONE(mode='local')
bwm_df = bwm_query()

all_trials = []
err = []
for eid in bwm_df['eid'].unique():
    try:
        sl = SessionLoader(one, eid)
        sl.load_trials()
        sl.trials.insert(0, 'eid', eid)
        all_trials.append(sl.trials)
    except BaseException as e:
        print(eid, e)
        err.append((eid, e))

df_trials = pd.concat(all_trials, ignore_index=True)

# Mark which trials pass bwm default trials qc
min_rt = 0.08
max_rt = 2
nan_exclude = [
    'stimOn_times',
    'choice',
    'feedback_times',
    'probabilityLeft',
    'firstMovement_times',
    'feedbackType'
]
query = f'(firstMovement_times - stimOn_times < {min_rt}) | (firstMovement_times - stimOn_times > {max_rt})'
for event in nan_exclude:
    query += f' | {event}.isnull()'
df_trials['bwm_include'] = df_trials.eval(query)

# Save to file
df_trials.to_parquet(local_path)

# Upload to s3
print(f'aws s3 sync "{local_path}" s3://ibl-brain-wide-map-private/aggregates/2022_Q4_IBL_et_al_BWM/trials.pqt')
