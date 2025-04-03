import hashlib
import numpy as np
import os
import pandas as pd
import unittest

from one.api import ONE

from brainwidemap import bwm_loading


class TestBWMLoading(unittest.TestCase):
    def setUp(self):
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.default_df = bwm_loading.bwm_query()

    def test_data_freeze(self):
        # Test default
        hashes = pd.util.hash_pandas_object(self.default_df)
        assert (hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                             ).hexdigest() == '78255dd2649024c2a51b26ac049758d4927a2c87')
        assert self.default_df.shape[0] == 699

        # Test explicit freezes
        df_bwm = bwm_loading.bwm_query(freeze='2022_10_initial')
        hashes = pd.util.hash_pandas_object(df_bwm)
        assert (hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                             ).hexdigest() == '56f88ec777f496bf8788973311e9610bcb21df0c')
        assert df_bwm.shape[0] == 552

        df_bwm = bwm_loading.bwm_query(freeze='2022_10_update')
        hashes = pd.util.hash_pandas_object(df_bwm)
        assert (hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                             ).hexdigest() == 'f69b71321059a1068d7306dc5b763a00d0e6a0c7')
        assert df_bwm.shape[0] == 557

        df_bwm = bwm_loading.bwm_query(freeze='2022_10_bwm_release')
        hashes = pd.util.hash_pandas_object(df_bwm)
        assert (hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                             ).hexdigest() == '4e05721092ed4ae1533eba77fc56817765cefda7')
        assert df_bwm.shape[0] == 547

        df_bwm = bwm_loading.bwm_query(freeze='2023_12_bwm_release')
        hashes = pd.util.hash_pandas_object(df_bwm)
        assert (hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                             ).hexdigest() == '78255dd2649024c2a51b26ac049758d4927a2c87')
        assert df_bwm.shape[0] == 699

    def test_spike_load_and_merge_probes(self):
        eid = self.default_df['eid'].iloc[99]
        tmp_df = self.default_df.set_index(['eid', 'subject']).xs(eid, level='eid')
        pids = tmp_df['pid'].to_list()
        probe_names = tmp_df['probe_name'].to_list()

        spikes_list = []
        clusters_list = []
        for pid, probe_name in zip(pids, probe_names):
            spikes, clusters = bwm_loading.load_good_units(self.one, pid, eid=eid, pname=probe_name)
            assert len(spikes['times']) == len(spikes['amps']) == len(spikes['clusters'] == len(spikes['depths']))

            spikes_list.append(spikes)
            clusters_list.append(clusters)

        merged_spikes, merged_clusters = bwm_loading.merge_probes(spikes_list, clusters_list)
        # Check that the spike arrays contain all individual arrays
        assert all([set(merged_spikes[k]) == set(np.concatenate([s[k] for s in spikes_list]))
                    for k in merged_spikes.keys()])
        # Check that the times are sorted in ascending order
        assert np.sum(np.diff(merged_spikes['times']) < 0) == 0
        # check that we have only good clusters
        assert np.all(merged_clusters['label'] == 1)
        # check that clusters were re-indexed and spikes['clusters'] updated accordingly
        assert np.sum(np.diff(merged_clusters.index) <= 0) == 0
        assert set(merged_spikes['clusters']).issubset(set(merged_clusters.index))
        assert merged_clusters.iloc[:len(clusters_list[0])].compare(clusters_list[0]).empty
        assert merged_clusters.iloc[len(clusters_list[0]):].reset_index(drop=True).compare(
            clusters_list[1][merged_clusters.columns]).empty
        idx = np.random.randint(0, merged_clusters.shape[0] - 1)
        if idx < clusters_list[0].shape[0]:
            assert merged_clusters.iloc[idx].compare(clusters_list[0].iloc[idx]).empty
        else:
            assert merged_clusters.iloc[idx].compare(
                clusters_list[1].iloc[idx - clusters_list[0].shape[0]][merged_clusters.columns]).empty

    def test_filter_units_region(self):

        # Test with downloading clusters table first
        clusters_table = bwm_loading.download_aggregate_tables(self.one, type='clusters', overwrite=True)
        assert clusters_table.exists()

        from iblutil.numerical import hash_uuids
        units_df = bwm_loading.filter_units_region(self.default_df['eid'], clusters_table=clusters_table)
        assert units_df.Beryl.nunique() == 210
        assert hash_uuids(units_df['uuids']) == 'd16d0b38d392b18c0ce8b615ec89d60d7c901df2eeb3432986b62130af28ef01'

        # Test without passing clusters table
        units_df = bwm_loading.filter_units_region(self.default_df['eid'], one=self.one)
        assert hash_uuids(units_df['uuids']) == 'd16d0b38d392b18c0ce8b615ec89d60d7c901df2eeb3432986b62130af28ef01'

        # Test QC filter only
        units_df = bwm_loading.filter_units_region(self.default_df['eid'], clusters_table=clusters_table, min_qc=1,
                                                   min_units_sessions=None)
        assert hash_uuids(units_df['uuids']) == 'ed80af64a83a055f049e4b9f57fdc3a0d135cb867d1eebb31073bd213ebb386c'

        # Test units filter only
        units_df = bwm_loading.filter_units_region(self.default_df['eid'], clusters_table=clusters_table, min_qc=None,
                                                   min_units_sessions=(5, 2))
        assert hash_uuids(units_df['uuids']) == '82a43bb2344a960b0f39a5c28fa56406dd5788b7946d0d178cb36205b1029b92'


    def test_filter_trials(self):

        # Test with downloading clusters table first
        trials_table = bwm_loading.download_aggregate_tables(self.one, type='trials', overwrite=True)
        assert trials_table.exists()

        eids = bwm_loading.filter_sessions(
            self.default_df['eid'], trials_table=trials_table, bwm_include=False, min_errors=None
        )
        assert len(eids) == 459

        eids = bwm_loading.filter_sessions(
            self.default_df['eid'], trials_table=trials_table, bwm_include=True, min_errors=None
        )
        assert len(eids) == 459

        eids = bwm_loading.filter_sessions(
            self.default_df['eid'], trials_table=trials_table, bwm_include=True, min_errors=3
        )
        assert len(eids) == 459

        eids = bwm_loading.filter_sessions(
            self.default_df['eid'], trials_table=trials_table, bwm_include=True, min_errors=3,
            saturation_intervals='saturation_stim_plus04'
        )
        assert len(eids) == 459

        eids = bwm_loading.filter_sessions(
            self.default_df['eid'], trials_table=trials_table, bwm_include=True, min_errors=3,
            saturation_intervals=[
                'saturation_stim_plus04',
                'saturation_feedback_plus04',
                'saturation_move_minus02',
                'saturation_stim_minus04_minus01',
                'saturation_stim_plus06',
                'saturation_stim_minus06_plus06',
                'saturation_stim_minus06_minus01'
            ]
        )
        assert len(eids) == 458

        trials_table.unlink()

    def test_trials_and_mask(self):
        # Test two different sessions with default settings
        eid_1 = '5569f363-0934-464e-9a5b-77c8e67791a1'
        eid_2 = 'dda5fc59-f09a-4256-9fb5-66c67667a466'
        trials, mask = bwm_loading.load_trials_and_mask(self.one, eid_1)
        assert mask.sum() == 513
        trials, mask = bwm_loading.load_trials_and_mask(self.one, eid_2)
        assert mask.sum() == 438
        # Test them with additional setting
        trials, mask = bwm_loading.load_trials_and_mask(self.one, eid_1, min_trial_len=0, max_trial_len=100,
                                                        exclude_nochoice=True, exclude_unbiased=True)
        assert mask.sum() == 455
        trials, mask = bwm_loading.load_trials_and_mask(self.one, eid_2, min_trial_len=0, max_trial_len=100,
                                                        exclude_nochoice=True, exclude_unbiased=True)
        assert mask.sum() == 395
        # Test with different saturation intervals
        trials, mask = bwm_loading.load_trials_and_mask(self.one, eid_1, saturation_intervals='saturation_stim_plus04')
        assert mask.sum() == 513
        trials, mask = bwm_loading.load_trials_and_mask(self.one, eid_2, nan_exclude=['choice'])
        assert mask.sum() == 441
        trials, mask = bwm_loading.load_trials_and_mask(
            self.one, eid_2, nan_exclude=['choice'],
            saturation_intervals=['saturation_stim_minus04_minus01', 'saturation_move_minus02'])
        assert mask.sum() == 438

    def test_video_filter(self):
        eids = list(self.default_df.eid.unique())

        for cam, num in zip(['left', 'right', 'body'], [435, 433, 257]):
            assert len(bwm_loading.filter_video_data(
                self.one, eids, camera=cam, min_video_qc='FAIL', min_dlc_qc='FAIL'
            )) == num

        for cam, num in zip(['left', 'right', 'body'], [436, 433, 257]):
            assert len(bwm_loading.filter_video_data(
                self.one, eids, camera=cam, min_video_qc='FAIL', min_dlc_qc=None
            )) == num

        for cam, num in zip(['left', 'right', 'body'], [435, 433, 258]):
            assert len(bwm_loading.filter_video_data(
                self.one, eids, camera=cam, min_video_qc=None, min_dlc_qc='FAIL'
            )) == num
