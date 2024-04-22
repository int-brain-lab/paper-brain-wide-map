""""
This is run on parede. If any of the data have been patched, you need to remove the cached results for the relevant pid
from the CACHE_DIR before running. To sync to AWS see the commands at the end of the script
"""

from pathlib import Path
import numpy as np
import pandas as pd
import urllib.error
from datetime import date
import tqdm

from one.api import ONE
from ibllib.atlas import AllenAtlas
from iblutil.util import setup_logger

from brainwidemap import bwm_query
from brainbox.io.one import SpikeSortingLoader

logger = setup_logger('brainbox')

ba = AllenAtlas()

year_week = date.today().isocalendar()[:2]
STAGING_PATH = Path('/mnt/s1/aggregates/2023_Q4_IBL_et_al_BWM_2').joinpath(f'{year_week[0]}_W{year_week[1]:02}_bwm')
CACHE_DIR = Path("/mnt/s0/bwm_julia")  # this is the path containing the metrics and clusters tables for fast releoading

excludes = []
errorkey = []
error404 = []
one = ONE()
bwm_df = bwm_query()
pids = bwm_df['pid']
# init dataframes
df_probes = pd.DataFrame(dict(eid='', pname='', spike_sorter='', histology=''), index=pids)
ldf_channels = []
ldf_clusters = []
ldf_depths = []
no_spike_sorting = []

IMIN = 0
for i, pid in tqdm.tqdm(enumerate(pids)):
    if i < IMIN:
        continue
    eid, pname = one.pid2eid(pid)
    df_probes['eid'][i] = eid
    df_probes['pname'][i] = pname
    logger.info(f"{i}/{len(pids)}, {pid}")
    ss = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    try:
        spikes, clusters, channels = ss.load_spike_sorting(query_type='remote')
    except urllib.error.HTTPError:
        error404.append(pid)
        logger.error(f"{pid} error 404")
        continue
    except KeyError:
        errorkey.append(pid)
        logger.error(f"{pid} key error")
        continue
    cache_dir_cluster = CACHE_DIR.joinpath(f"{pid}")
    cache_dir_cluster.mkdir(exist_ok=True)
    if cache_dir_cluster.joinpath('clusters.pqt').exists():
        df_clusters = pd.read_parquet(cache_dir_cluster.joinpath('clusters.pqt'))
        _clusters = {}
        for k in df_clusters.keys():
            _clusters[k] = df_clusters[k].values
        clusters = _clusters
    else:
        clusters = ss.merge_clusters(spikes, clusters, channels, compute_metrics=False)
    df_probes['spike_sorter'][i] = ss.collection
    df_probes['histology'][i] = ss.histology
    if not spikes:
        no_spike_sorting.append(pid)
        continue
    df_ch = pd.DataFrame(channels)
    df_ch['pid'] = pid
    ldf_channels.append(df_ch)
    df_clu = pd.DataFrame(clusters)
    df_clu['pid'] = pid
    df_clu['eid'] = eid
    ldf_clusters.append(df_clu)
    # aggregate spike features per depth
    df_spikes = pd.DataFrame(spikes)
    df_spikes.dropna(axis=0, how='any', inplace=True)
    df_spikes['rdepths'] = (np.round(df_spikes['depths'] / 20) * 20).astype(np.int32)
    df_spikes['amps'] = df_spikes['amps'] * 1e6
    df_depths = df_spikes.groupby('rdepths').agg(
        amps=pd.NamedAgg(column="amps", aggfunc="median"),
        amps_std=pd.NamedAgg(column="amps", aggfunc="std"),
        cell_count=pd.NamedAgg(column="clusters", aggfunc="nunique"),
        spike_rate=pd.NamedAgg(column="amps", aggfunc="count"),
    )
    df_depths['pid'] = pid
    df_depths['spike_rate'] = df_depths['spike_rate'] / (np.max(spikes['times']) - np.min(spikes['times']))
    ldf_depths.append(df_depths)

df_channels = pd.concat(ldf_channels, ignore_index=True)
df_clusters = pd.concat(ldf_clusters, ignore_index=True)
df_depths = pd.concat(ldf_depths)

# todo: compute the df_channels['raw_ind'] field by modifying the spike sorting loader behaviour
# convert the depths dataframe to a multi-index dataframe
df_depths['depths'] = df_depths.index.values
df_depths = df_depths.set_index(['pid', 'depths'])

# saves the 3 dataframes
STAGING_PATH.mkdir(exist_ok=True, parents=True)
df_channels.to_parquet(STAGING_PATH.joinpath('channels.pqt'))
df_clusters.to_parquet(STAGING_PATH.joinpath('clusters.pqt'))
df_probes.to_parquet(STAGING_PATH.joinpath('probes.pqt'))
df_depths.to_parquet(STAGING_PATH.joinpath('depths.pqt'))

print(f'cp {STAGING_PATH.joinpath("*")} {STAGING_PATH.parent}')
print(f'aws s3 sync "{STAGING_PATH.parent}" s3://ibl-brain-wide-map-private/aggregates/2023_Q4_IBL_et_al_BWM_2 --profile ibl')
print(errorkey)
print(error404)

# On SDSC, sync from S3 (might have to adjust the paths if you are working with a different tag)
# aws s3 sync s3://ibl-brain-wide-map-private/aggregates/2023_Q4_IBL_et_al_BWM_2 /mnt/ibl/aggregates/2023_Q4_IBL_et_al_BWM_2/ --profile ibladmin

# If you are ready to make this public, copy on SDSC to the public folder and sync to public S3
# cp /mnt/ibl/aggregates/2023_Q4_IBL_et_al_BWM_2/clusters.pqt /mnt/ibl/public/aggregates/2023_Q4_IBL_et_al_BWM_2/
# aws s3 sync /mnt/ibl/public/aggregates/2023_Q4_IBL_et_al_BWM_2/ s3://ibl-brain-wide-map-public/aggregates/2023_Q4_IBL_et_al_BWM_2  --profile ibladmin
