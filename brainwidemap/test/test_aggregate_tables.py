import numpy as np
import pandas as pd
import unittest

from one.api import ONE

from brainwidemap import bwm_loading


class TestAggregateTables(unittest.TestCase):

    def setUp(self):
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.trials_table = bwm_loading.download_aggregate_tables(self.one, type='trials')
        self.df_trials = pd.read_parquet(self.trials_table)

    def test_bwm_logic_true(self):
        for i in range(self.df_trials.shape[0]):
            if self.df_trials.iloc[i]['bwm_include'] is True:
                assert self.df_trials.iloc[i]['firstMovement_times'] - self.df_trials.iloc[i]['stimOn_times'] >= 0.08
                assert self.df_trials.iloc[i]['firstMovement_times'] - self.df_trials.iloc[i]['stimOn_times'] <= 2
                assert np.isnan(self.df_trials.iloc[i]['stimOn_times']) is False
                assert np.isnan(self.df_trials.iloc[i]['choice']) is False
                assert np.isnan(self.df_trials.iloc[i]['feedback_times']) is False
                assert np.isnan(self.df_trials.iloc[i]['probabilityLeft']) is False
                assert np.isnan(self.df_trials.iloc[i]['firstMovement_times']) is False
                assert np.isnan(self.df_trials.iloc[i]['feedbackType']) is False

    def test_bwm_logic_false(self):
        for i in range(self.df_trials.shape[0]):
            if self.df_trials.iloc[i]['bwm_include'] is False:
                assert ((self.df_trials.iloc[i]['firstMovement_times'] - self.df_trials.iloc[i]['stimOn_times'] < 0.08) or
                        (self.df_trials.iloc[i]['firstMovement_times'] - self.df_trials.iloc[i]['stimOn_times'] > 2) or
                        (np.isnan(self.df_trials.iloc[i]['stimOn_times']) is True) or
                        (np.isnan(self.df_trials.iloc[i]['choice']) is True) or
                        (np.isnan(self.df_trials.iloc[i]['feedback_times']) is True) or
                        (np.isnan(self.df_trials.iloc[i]['probabilityLeft']) is True) or
                        (np.isnan(self.df_trials.iloc[i]['firstMovement_times']) is True) or
                        (np.isnan(self.df_trials.iloc[i]['feedbackType']) is True))

    def tearDown(self):
        self.trials_table.unlink()
