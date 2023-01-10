import hashlib
import pandas as pd
import numpy as np

from one.api import ONE
from brainwidemap import bwm_loading


def test_data_freeze():
    df_bwm = bwm_loading.bwm_query(freeze='2022_10_initial')
    hashes = pd.util.hash_pandas_object(df_bwm)
    assert(hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                        ).hexdigest() == '56f88ec777f496bf8788973311e9610bcb21df0c')
    assert df_bwm.shape[0] == 552

    df_bwm = bwm_loading.bwm_query(freeze='2022_10_update')
    hashes = pd.util.hash_pandas_object(df_bwm)
    assert(hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                        ).hexdigest() == 'f69b71321059a1068d7306dc5b763a00d0e6a0c7')
    assert df_bwm.shape[0] == 557

    df_bwm = bwm_loading.bwm_query(freeze='2022_10_bwm_release')
    hashes = pd.util.hash_pandas_object(df_bwm)
    assert(hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                        ).hexdigest() == '4e05721092ed4ae1533eba77fc56817765cefda7')
    assert df_bwm.shape[0] == 547

    # Test default
    df_bwm = bwm_loading.bwm_query()
    hashes = pd.util.hash_pandas_object(df_bwm)
    assert(hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                        ).hexdigest() == '4e05721092ed4ae1533eba77fc56817765cefda7')
    assert df_bwm.shape[0] == 547


def test_spike_load_and_merge_probes():
    one = ONE()
    df_bwm = bwm_loading.bwm_query()
    eid = df_bwm['eid'].iloc[99]
    tmp_df = df_bwm.set_index(['eid', 'subject']).xs(eid, level='eid')
    pids = tmp_df['pid'].to_list()
    probe_names = tmp_df['probe_name'].to_list()

    spikes_list = []
    clusters_list = []
    for pid, probe_name in zip(pids, probe_names):
        spikes, clusters = bwm_loading.load_good_units(one, pid, eid=eid, pname=probe_name)
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


def test_filter_regions():
    one = ONE()
    bwm_df = bwm_loading.bwm_query()

    # Test with downloading clusters table first
    clusters_table = bwm_loading.download_aggregate_tables(one, type='clusters')
    assert clusters_table.exists()

    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table)
    assert set(regions_df.keys()) == set(['Beryl', 'pid', 'n_units', 'n_probes', 'n_sessions'])
    assert regions_df.shape == (2469, 5)

    # Test without passing clusters table
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], one=one)
    assert regions_df.shape == (2469, 5)

    # Test QC filter only
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table, min_qc=1,
                                            min_units_region=None, min_probes_region=None)
    assert regions_df.shape == (2569, 5)

    # Test units filter only
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table, min_qc=None,
                                            min_units_region=10, min_probes_region=None)
    assert regions_df.shape == (3094, 5)

    # Test probes filter only
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table, min_qc=None,
                                            min_units_region=None, min_probes_region=2)
    assert regions_df.shape == (3087, 5)

    # Test session filter
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table,
                                            min_probes_region=None, min_sessions_region=2)
    assert regions_df.shape == (2469, 5)

    # Remove the table
    clusters_table.unlink()


def test_filter_trials():
    one = ONE()
    bwm_df = bwm_loading.bwm_query()

    # Test with downloading clusters table first
    trials_table = bwm_loading.download_aggregate_tables(one, type='trials')
    assert trials_table.exists()

    eids = bwm_loading.filter_sessions(bwm_df['eid'], trials_table=trials_table)
    assert eids.shape == (345,)

    eids = bwm_loading.filter_sessions(bwm_df['eid'], one=one, min_trials=None)
    assert eids.shape == (354,)

    trials_table.unlink()


def test_trials_and_mask():
    one = ONE()
    # Test two different sessions with default settings
    eid_1 = '5569f363-0934-464e-9a5b-77c8e67791a1'
    eid_2 = 'dda5fc59-f09a-4256-9fb5-66c67667a466'
    trials, mask = bwm_loading.load_trials_and_mask(one, eid_1)
    assert mask.sum() == 513
    trials, mask = bwm_loading.load_trials_and_mask(one, eid_2)
    assert mask.sum() == 438
    # Test them with additional setting
    trials, mask = bwm_loading.load_trials_and_mask(one, eid_1, min_trial_len=0, max_trial_len=100,
                                                    exclude_nochoice=True, exclude_unbiased=True)
    assert mask.sum() == 455
    trials, mask = bwm_loading.load_trials_and_mask(one, eid_2, min_trial_len=0, max_trial_len=100,
                                                    exclude_nochoice=True, exclude_unbiased=True)
    assert mask.sum() == 395
