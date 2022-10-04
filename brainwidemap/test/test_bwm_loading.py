import hashlib
import pandas as pd
import numpy as np

from one.api import ONE
from brainwidemap import bwm_loading


def test_data_freeze():
    df_bwm = bwm_loading.bwm_query()
    hashes = pd.util.hash_pandas_object(df_bwm)
    assert(hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                        ).hexdigest() == '56f88ec777f496bf8788973311e9610bcb21df0c')
    assert df_bwm.shape[0] == 552

    df_bwm = bwm_loading.bwm_query(freeze='2022_10_initial')
    hashes = pd.util.hash_pandas_object(df_bwm)
    assert(hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                        ).hexdigest() == '56f88ec777f496bf8788973311e9610bcb21df0c')
    assert df_bwm.shape[0] == 552


def test_aggregates_and_filter():
    one = ONE()
    df_bwm = bwm_loading.bwm_query(freeze='2022_10_initial')

    # Test download trials and clusters table
    trials_table = bwm_loading.download_aggregate_tables(one, type='trials')
    clusters_table = bwm_loading.download_aggregate_tables(one, type='clusters')
    assert trials_table.exists()
    assert clusters_table.exists()
    # Test filter without passing trials and clusters table
    df_bwm_filter = bwm_loading.bwm_filter(bwm_df=df_bwm, one=one)
    assert df_bwm_filter.shape == (230, 7)
    # Test with passing trials and clusters tablbe
    df_bwm_filter = bwm_loading.bwm_filter(bwm_df=df_bwm, clusters_table=clusters_table, trials_table=trials_table)
    assert df_bwm_filter.shape == (230, 7)
    # Test trials filter only
    df_bwm_filter_trials = bwm_loading.bwm_filter(bwm_df=df_bwm, trials_table=trials_table, min_trials=200, min_qc=None,
                                                  min_units_region=None, min_probes_region=None)
    assert df_bwm_filter_trials.shape == (245, 7)
    # Test QC filter only
    df_bwm_filter_qc = bwm_loading.bwm_filter(bwm_df=df_bwm, clusters_table=clusters_table, min_trials=None, min_qc=1,
                                              min_units_region=None, min_probes_region=None)
    assert df_bwm_filter_qc.shape == (523, 7)
    # Test units filter only
    df_bwm_filter_units = bwm_loading.bwm_filter(bwm_df=df_bwm, clusters_table=clusters_table, min_trials=None,
                                                 min_qc=None, min_units_region=10, min_probes_region=None)
    assert df_bwm_filter_units.shape == (524, 7)
    # Test probes filter only
    df_bwm_filter_probes = bwm_loading.bwm_filter(bwm_df=df_bwm, clusters_table=clusters_table, min_trials=None,
                                                  min_qc=None, min_units_region=None, min_probes_region=2)
    assert df_bwm_filter_probes.shape == (524, 7)

    # Remove the tables
    trials_table.unlink()
    clusters_table.unlink()


