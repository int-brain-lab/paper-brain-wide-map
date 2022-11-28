from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import (
    bwm_query, load_good_units, load_trials_and_mask, filter_regions, filter_sessions,
    download_aggregate_tables)
import sys

N_PARA = 50
para_index = int(sys.argv[1])-1

"""
-----------
USER INPUTS
-----------
"""
# Whether to remove data based on number of probes and units per region, good trials (see below for specifics)
regions_filter = True
trials_filter = True
# Whether to download wheel and whisker data as well
wheel_data = True
whisker_data = True
# You can choose the cache directory if you want it to be different from the default
cache_dir = None  # Path('/full/path/to/cache')

"""
----------
CACHE DATA
-----------
"""
# For caching, we use an online instance.
if cache_dir:
    one = ONE(cache_dir=cache_dir, base_url='https://alyx.internationalbrainlab.org', mode='remote')
else:
    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    cache_dir = one.cache_dir

BAD_PIDS = ['367ea4c4-d1b7-47a6-9f18-5d0df9a3f4de',
            '9b3ad89a-177f-4242-9a96-2fd98721e47f',
            '485b50c8-71e1-4654-9a07-64395c15f5ed',
            '0aafb6f1-6c10-4886-8f03-543988e02d9e',
            'b976e816-bc24-42e3-adf4-2801f1a52657',
            'dac5defe-62b8-4cc0-96cb-b9f923957c7f',
            'e9cf749b-85dc-4b59-834b-325cec608c48',
            'f84f36c9-88f8-4d80-ba34-7f7a2b254ece',
            '96c816ad-9a48-46a4-8a84-9a73cc153d69']
BAD_PIDS = []
BAD_EIDS = [one.pid2eid(p)[0] for p in BAD_PIDS]

# Get the dataframe of all included sessions and probe insertions, use frozen query
bwm_df = bwm_query(freeze='2022_10_bwm_release')

if regions_filter:
    # Download the latest clusters table, we use the same cache as above
    clusters_table = download_aggregate_tables(one, type='clusters', target_path=cache_dir)
    # Map probes to regions (here: Beryl mapping) and filter according to QC, number of units and probes per region
    region_df = filter_regions(bwm_df['pid'], clusters_table=clusters_table, mapping='Beryl', min_qc=1,
                               min_units_region=10, min_probes_region=None, min_sessions_region=2)
    # Remove probes and sessions based on this filters
    bwm_df = bwm_df[bwm_df['pid'].isin(region_df['pid'].unique())]

if trials_filter:
    # Download the latest trials table and filter for sessions that have at least 200 trials fulfilling BWM criteria
    trials_table = download_aggregate_tables(one, type='trials', target_path=cache_dir)
    eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, min_trials=200)
    bwm_df = bwm_df[bwm_df['eid'].isin(eids)]

# Download cluster information and spike trains for all good units.
count = -1
for pid in bwm_df['pid']:
    count += 1
    if not (count%N_PARA == para_index):
        continue
    elif pid in BAD_PIDS:
        continue
    else:
       print(f"Downloading spike sorting data for {pid}")
       spikes, clusters = load_good_units(one, pid, compute_metrics=False)

# Download trials for all sessions
count = -1
for eid in bwm_df['eid']:
    count += 1
    if not (count%N_PARA == para_index):
        continue
    else: # Oddly not problem with errors here, dont need BAD_EIDS like wheel or spike sorting
        print(f"Downloading trials data for {eid}")
        sess_loader = SessionLoader(one, eid)
        sess_loader.load_trials()

# Download wheel data for all sessions
if wheel_data:
    count = -1
    for eid in bwm_df['eid']:
        count += 1
        if not (count%N_PARA == para_index):
       	    continue

        #elif eid == 'cc45c568-c3b9-4f74-836e-c87762e898c8' or (eid in BAD_EIDS):
        #    continue
        else:
            print(f"Downloading wheel data for {eid}")
            sess_loader = SessionLoader(one, eid)
            sess_loader.load_wheel()

# Download whisker data for all sessions
if whisker_data:
    count = -1
    me_err = []
    for eid in bwm_df['eid']:
        count += 1
        if not (count%N_PARA == para_index):
       	    continue
        else:
            try:
                print(f"Downloading motion energy data for {eid}")
                sess_loader = SessionLoader(one, eid)
                sess_loader.load_motion_energy(views=['left', 'right'])
            except BaseException as e:
                print(eid, e)
                me_err.append((eid, e))
