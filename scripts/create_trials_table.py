from datetime import date
import pandas as pd
from pathlib import Path

from one.api import ONE
from brainwidemap import bwm_query, load_trials_and_mask


year_week = date.today().isocalendar()[:2]
STAGING_PATH = Path('/mnt/s1/aggregates/2023_Q4_IBL_et_al_BWM_2').joinpath(f'{year_week[0]}_W{year_week[1]:02}_bwm')


one = ONE(base_url='https://openalyx.internationalbrainlab.org')
bwm_df = bwm_query()

all_trials = []
err = []
for i, eid in enumerate(bwm_df['eid'].unique()):
    try:
        print(f'Session {i}/{len(bwm_df["eid"].unique())}')
        trials, mask = load_trials_and_mask(one, eid)
        trials.insert(0, 'eid', eid)
        trials['bwm_include'] = mask
        all_trials.append(trials)
    except BaseException as e:
        print(eid, e)
        err.append((eid, e))

df_trials = pd.concat(all_trials, ignore_index=True)

# Save to file
STAGING_PATH.mkdir(exist_ok=True, parents=True)
df_trials.to_parquet(STAGING_PATH.joinpath('trials.pqt'))

# Upload to s3
week_file = STAGING_PATH.joinpath('trials.pqt')
root_file = STAGING_PATH.parent.joinpath('trials.pqt')
print(f"cp {week_file} {root_file}")
print(f'aws s3 sync "{STAGING_PATH.parent}" s3://ibl-brain-wide-map-private/aggregates/2023_Q4_IBL_et_al_BWM_2/ --profile ibl')
