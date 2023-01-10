import numpy as np
import pandas as pd

from one.api import ONE
from brainwidemap import bwm_loading

one = ONE()
trials_table = bwm_loading.download_aggregate_tables(one, type='trials')
df_trials = pd.read_parquet(trials_table)

for i in range(df_trials.shape[0]):
    if df_trials.iloc[i]['bwm_include'] is False:
        assert ((df_trials.iloc[i]['firstMovement_times'] - df_trials.iloc[i]['stimOn_times'] < 0.08) or
                (df_trials.iloc[i]['firstMovement_times'] - df_trials.iloc[i]['stimOn_times'] > 2) or
                (np.isnan(df_trials.iloc[i]['stimOn_times']) is True) or
                (np.isnan(df_trials.iloc[i]['choice']) is True) or
                (np.isnan(df_trials.iloc[i]['feedback_times']) is True) or
                (np.isnan(df_trials.iloc[i]['probabilityLeft']) is True) or
                (np.isnan(df_trials.iloc[i]['firstMovement_times']) is True) or
                (np.isnan(df_trials.iloc[i]['feedbackType']) is True))


for i in range(df_trials.shape[0]):
    if df_trials.iloc[i]['bwm_include'] is True:
        assert df_trials.iloc[i]['firstMovement_times'] - df_trials.iloc[i]['stimOn_times'] >= 0.08
        assert df_trials.iloc[i]['firstMovement_times'] - df_trials.iloc[i]['stimOn_times'] <= 2
        assert np.isnan(df_trials.iloc[i]['stimOn_times']) is False
        assert np.isnan(df_trials.iloc[i]['choice']) is False
        assert np.isnan(df_trials.iloc[i]['feedback_times']) is False
        assert np.isnan(df_trials.iloc[i]['probabilityLeft']) is False
        assert np.isnan(df_trials.iloc[i]['firstMovement_times']) is False
        assert np.isnan(df_trials.iloc[i]['feedbackType']) is False