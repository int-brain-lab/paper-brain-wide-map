from datetime import date
import pandas as pd
from pathlib import Path

from one.api import ONE
from brainwidemap import bwm_query
from brainbox.io.one import SessionLoader


year_week = date.today().isocalendar()[:2]
STAGING_PATH = Path('/mnt/s0/aggregates/2022_Q4_IBL_et_al_BWM').joinpath(f'{year_week[0]}_W{year_week[1]:02}_bwm')

one = ONE(base_url='https://alyx.internationalbrainlab.org')
bwm_df = bwm_query()

all_trials = []
err = []
for i, eid in enumerate(bwm_df['eid'].unique()):
    try:
        print(f'Session {i}/{len(bwm_df["eid"].unique())}')
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
df_trials.to_parquet(STAGING_PATH.joinpath('trials.pqt'))

# Upload to s3
week_file = STAGING_PATH.joinpath('trials.pqt')
root_file = STAGING_PATH.parent.joinpath('trials.pqt')
week = STAGING_PATH.name
print(f"cp {week_file} {root_file}")
print(f'aws s3 sync "{week_file}" s3://ibl-brain-wide-map-private/aggregates/2022_Q4_IBL_et_al_BWM/trials.pqt')
print(f'aws s3 sync "{week_file}" s3://ibl-brain-wide-map-private/aggregates/2022_Q4_IBL_et_al_BWM/{week}/trials.pqt')
