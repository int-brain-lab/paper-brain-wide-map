from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import (
    bwm_query, load_good_units, load_trials_and_mask, filter_regions, filter_sessions,
    download_aggregate_tables)
import sys

# number of jobs which should be submited by a .sh file to parallelize data caching
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
min_trials = 150
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

BAD_PIDS = [] # can exclude a list of PIDS here, but currently not necessary
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
    eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, min_trials=min_trials)
    bwm_df = bwm_df[bwm_df['eid'].isin(eids)]

# Download cluster information and spike trains for all good units.
for count, pid in enumerate(bwm_df['pid']):
    if not (count%N_PARA == para_index):
        continue
    elif pid in BAD_PIDS:
        continue
    else:
       print(f"Downloading spike sorting data for {pid}")
       spikes, clusters = load_good_units(one, pid, compute_metrics=False)

# Download trials for all sessions
for count, eid in enumerate(bwm_df['eid']):
    if not (count%N_PARA == para_index):
        continue
    else: # Oddly not problem with errors here, dont need BAD_EIDS like wheel or spike sorting
        print(f"Downloading trials data for {eid}")
        sess_loader = SessionLoader(one, eid)
        sess_loader.load_trials()

# Download wheel data for all sessions
if wheel_data:
    for count, eid in enumerate(bwm_df['eid']):
        if not (count%N_PARA == para_index):
       	    continue
        # the following eid, cc45c568..., produces "RuntimeWarning: Failed to connect"
        elif eid == 'cc45c568-c3b9-4f74-836e-c87762e898c8' or (eid in BAD_EIDS):
            continue
        else:
            print(f"Downloading wheel data for {eid}")
            sess_loader = SessionLoader(one, eid)
            sess_loader.load_wheel()

# Download whisker data for all sessions
if whisker_data:
    me_err = []
    for count, eid in enumerate(bwm_df['eid']):
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
