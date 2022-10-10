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


def test_filter_regions():
    one = ONE()
    bwm_df = bwm_loading.bwm_query(freeze='2022_10_initial')

    # Test with downloading clusters table first
    clusters_table = bwm_loading.download_aggregate_tables(one, type='clusters')
    assert clusters_table.exists()

    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table)
    assert set(regions_df.keys()) == set(['Beryl', 'pid', 'n_units', 'n_probes', 'n_sessions'])
    assert regions_df.shape == (2465, 5)

    # Test without passing clusters table
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], one=one)
    assert regions_df.shape == (2465, 5)

    # Test QC filter only
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table, min_qc=1,
                                            min_units_region=None, min_probes_region=None)
    assert regions_df.shape == (2574, 5)

    # Test units filter only
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table, min_qc=None,
                                            min_units_region=10, min_probes_region=None)
    assert regions_df.shape == (3105, 5)

    # Test probes filter only
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table, min_qc=None,
                                            min_units_region=None, min_probes_region=2)
    assert regions_df.shape == (3100, 5)

    # Test session filter
    regions_df = bwm_loading.filter_regions(bwm_df['pid'], clusters_table=clusters_table,
                                            min_probes_region=None, min_sessions_region=2)
    assert regions_df.shape == (2465, 5)

    # Remove the table
    clusters_table.unlink()


def test_filter_trials():
    one = ONE()
    bwm_df = bwm_loading.bwm_query(freeze='2022_10_initial')

    # Test with downloading clusters table first
    trials_table = bwm_loading.download_aggregate_tables(one, type='trials')
    assert trials_table.exists()

    eids = bwm_loading.filter_sessions(bwm_df['eid'], trials_table=trials_table)
    assert eids.shape == (155,)

    eids = bwm_loading.filter_sessions(bwm_df['eid'], one=one, min_trials=None)
    assert eids.shape == (350,)

    trials_table.unlink()
