from datetime import date
import pandas as pd
from pathlib import Path
import numpy as np

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
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

# Add saturation info
# Definition of windows of interest
trial_epochs = {
    'saturation_stim_plus04': {'event': 'stimOn_times', 'twin': [0, 0.4]},
    'saturation_feedback_plus04': {'event': 'feedback_times', 'twin': [0, 0.4]},
    'saturation_move_minus02': {'event': 'firstMovement_times', 'twin': [-0.2, 0]},
    'saturation_stim_minus04_minus01': {'event': 'stimOn_times', 'twin': [-0.4, -0.1]},
    'saturation_stim_plus06': {'event': 'stimOn_times', 'twin': [0, 0.6]},
    'saturation_stim_minus06_plus06': {'event': 'stimOn_times', 'twin': [-0.6, 0.6]},
    'saturation_stim_minus06_minus01': {'event': 'stimOn_times', 'twin': [-0.6, -0.1]}
}

for key in trial_epochs.keys():
    df_trials[key] = False

# Find the sessions that have saturation intervals
licks_dir = Path('/mnt/s1/licks')
interval_files = licks_dir.rglob('lickIntervals.npy')

for interval_file in interval_files:
    lick_interval_samples = np.load(interval_file)
    pid = interval_file.parent.name
    if pid not in bwm_df['pid'].tolist():
        continue
    eid = one.pid2eid(pid)[0]

    ssl = SpikeSortingLoader(one=one, pid=pid)
    trials = one.load_object(eid, 'trials')

    # These are the start and end times of the saturation intervals in the trials clock
    lick_intervals = ssl.samples2times(lick_interval_samples)

    for key, info in trial_epochs.items():
        trial_events = trials[info['event']]
        trial_intervals = np.vstack((trial_events + info['twin'][0], trial_events + info['twin'][1])).T
        # Find the trial intervals that overlap with saturation intervals
        overlaps = np.zeros(shape=trial_intervals.shape[0], dtype=bool)
        for i, trial_interval in enumerate(trial_intervals):
            if np.any(np.isnan(trial_interval)):
                overlaps[i] = np.nan
            # We just need to check if any lick start or any lick end lies with the trial interval, we don't
            # actually care which
            else:
                overlap = np.any(
                    (lick_intervals.flatten() > trial_interval[0]) & (lick_intervals.flatten() < trial_interval[1])
                )
                overlaps[i] = overlap

        # To account for some sessions being truncated to pass, we might need to cut off some trials
        n_trials = df_trials.loc[df_trials['eid'] == eid].shape[0]
        df_trials.loc[df_trials['eid'] == eid, key] = overlaps[:n_trials]

# Save to file
STAGING_PATH.mkdir(exist_ok=True, parents=True)
df_trials.to_parquet(STAGING_PATH.joinpath('trials.pqt'))

# Upload to s3
week_file = STAGING_PATH.joinpath('trials.pqt')
root_file = STAGING_PATH.parent.joinpath('trials.pqt')
print(f"cp {week_file} {root_file}")
print(f'aws s3 sync "{STAGING_PATH.parent}" s3://ibl-brain-wide-map-private/aggregates/2023_Q4_IBL_et_al_BWM_2/ --profile ibl')


# On SDSC, sync from S3 (might have to adjust the paths if you are working with a different tag)
# aws s3 sync s3://ibl-brain-wide-map-private/aggregates/2023_Q4_IBL_et_al_BWM_2 /mnt/ibl/aggregates/2023_Q4_IBL_et_al_BWM_2/ --profile ibladmin

# If you are ready to make this public, copy on SDSC to the public folder and sync to public S3
# cp /mnt/ibl/aggregates/2023_Q4_IBL_et_al_BWM_2/trials.pqt /mnt/ibl/public/aggregates/2023_Q4_IBL_et_al_BWM_2/
# aws s3 sync /mnt/ibl/public/aggregates/2023_Q4_IBL_et_al_BWM_2/ s3://ibl-brain-wide-map-public/aggregates/2023_Q4_IBL_et_al_BWM_2  --profile ibladmin